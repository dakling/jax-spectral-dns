#!/usr/bin/env python3

import jax
import jax.numpy as jnp

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    if hasattr(sys, 'ps1'):
        pass
from domain import PhysicalDomain

try:
    reload(sys.modules["field"])
except:
    if hasattr(sys, 'ps1'):
        pass
from field import PhysicalField

try:
    reload(sys.modules["equation"])
except:
    if hasattr(sys, 'ps1'):
        pass
from equation import Equation


class Heat_Eq(Equation):
    name = "heat equation"

    def __init__(self, domain, *fields):
        super().__init__(domain, *fields)
        u = self.get_initial_field("u")
        u.update_boundary_conditions()

    def perform_explicit_euler_step(self, dt, i):
        u = self.get_latest_field("u")
        eq = u.laplacian()
        new_u = u + eq * dt
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        # self.fields["u"].append(new_u)
        self.fields["u"][-1] = new_u
        return self.fields["u"]

    def perform_implicit_euler_step(self, dt, i):
        # only for 1D!
        u = self.get_latest_field("u")
        D2 = self.domain.get_cheb_mat_2_homogeneous_dirichlet(0)
        I = jnp.eye(D2.shape[0])
        new_u = PhysicalField(u.domain, jnp.linalg.inv(I - dt * D2) @ u.data, name=u.name)
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        # self.fields["u"].append(new_u)
        self.fields["u"][-1] = new_u
        return self.fields["u"]

    def perform_time_step(self, dt, i):
        return self.perform_explicit_euler_step(dt, i)
        # return self.perform_implicit_euler_step(dt, i)

    def solve(self, dt, number_of_steps):
        for i in jnp.arange(1, number_of_steps + 1):
            self.perform_time_step(dt, i)
        return self.fields["u"]

    def plot(self):
        if self.domain.number_of_dimensions <= 2:
            u_0 = self.get_initial_field("u")
            u_fin = self.get_latest_field("u")
            u_0.plot(u_fin)
        elif self.domain.number_of_dimensions == 3:
            u_0 = self.get_initial_field("u")
            u_fin = self.get_latest_field("u")
            for i in self.all_dimensions():
                u_0.plot_center(i, u_fin)


def solve_heat_eq_1D():
    # Nx = 100000
    Nx = 100

    domain = PhysicalDomain.create((Nx,), (False,))

    u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")
    heat_eq = Heat_Eq(domain, u)

    Nt = 3000
    dt = 5e-9
    # def before_time_step(heat_eq):
    #     print("step", heat_eq.time_step)
    # heat_eq.before_time_step_fn = before_time_step
    heat_eq.solve(dt, Nt)
    heat_eq.plot()
    v_final = heat_eq.get_latest_field("u")
    e0 = u.energy()
    print(e0)
    print(v_final.energy())

def optimize_heat_eq_1D():
    Nx = 100
    def run(v0):
        domain = PhysicalDomain.create((Nx,), (False,))

        u = PhysicalField(domain, v0, name="u")
        e0 = u.energy()

        heat_eq = Heat_Eq(domain, u)

        Nt = 3000
        dt = 5e-9
        heat_eq.solve(dt, Nt)
        v_final = heat_eq.get_latest_field("u")
        return v_final.energy() / e0



    domain = PhysicalDomain.create((Nx,), (False,))

    u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    u_0 = PhysicalField.FromFunc(domain, func=u_fn, name="u")

    v0s = [u_0.data]
    step_size = 1e-0
    sq_grad_sums = 0.0 * u_0.data
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_arr = jnp.array(corr)
        # corr_field = VectorField([Field(v0_0.domain, corr[i], name="correction_" + "xyz"[i]) for i in range(3)])
        # corr_field.plot_3d(2)
        print("gain: " + str(gain))
        # print("corr (abs): " + str(abs(corr_field)))
        sq_grad_sums += corr_arr**2.0
        # alpha = jnp.array([eps / (1e-10 + jnp.sqrt(sq_grad_sums[i])) for i in range(v0_0[0].data.shape)])
        # eps = step_size / ((1 + 1e-10) * jnp.sqrt(sq_grad_sums))
        eps = step_size

        v0s.append(v0s[-1] + eps * corr_arr)
        v0_new = PhysicalField(domain, v0s[-1])
        v0_new.set_name("u_0_" + str(i))
        v0_new.plot()

def solve_heat_eq_2D():
    Nx = 24
    Ny = Nx

    domain = PhysicalDomain.create((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")

    heat_eq = Heat_Eq(domain, u)

    Nt = 5000
    dt = 5e-5
    heat_eq.solve(dt, Nt)
    heat_eq.plot()


def solve_heat_eq_3D():
    Nx = 100
    Ny = 100
    Nz = 100

    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")
    heat_eq = Heat_Eq(domain, u)

    Nt = 3000
    dt = 5e-9
    # def before_time_step(heat_eq):
    #     print("step", heat_eq.time_step)
    # heat_eq.before_time_step_fn = before_time_step
    heat_eq.solve(dt, Nt)
    heat_eq.plot()
    v_final = heat_eq.get_latest_field("u")
    e0 = u.energy()
    print(e0)
    print(v_final.energy())

def optimize_heat_eq_3D():
    Nx = 100
    Ny = Nx
    Nz = Nx
    def run(v0):
        domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))

        u = PhysicalField(domain, v0, name="u")
        e0 = u.energy()

        heat_eq = Heat_Eq(domain, u)

        Nt = 3000
        dt = 5e-9
        heat_eq.solve(dt, Nt)
        v_final = heat_eq.get_latest_field("u")
        return v_final.energy() / e0



    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u_0 = PhysicalField.FromFunc(domain, func=u_fn, name="u")

    v0s = [u_0.data]
    step_size = 1e-0
    sq_grad_sums = 0.0 * u_0.data
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_arr = jnp.array(corr)
        # corr_field = VectorField([Field(v0_0.domain, corr[i], name="correction_" + "xyz"[i]) for i in range(3)])
        # corr_field.plot_3d(2)
        print("gain: " + str(gain))
        # print("corr (abs): " + str(abs(corr_field)))
        sq_grad_sums += corr_arr**2.0
        # alpha = jnp.array([eps / (1e-10 + jnp.sqrt(sq_grad_sums[i])) for i in range(v0_0[0].data.shape)])
        # eps = step_size / ((1 + 1e-10) * jnp.sqrt(sq_grad_sums))
        eps = step_size

        v0s.append(v0s[-1] + eps * corr_arr)
        v0_new = PhysicalField(domain, v0s[-1])
        v0_new.set_name("u_0_" + str(i))
        v0_new.plot_3d(2)

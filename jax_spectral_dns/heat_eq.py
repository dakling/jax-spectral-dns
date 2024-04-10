#!/usr/bin/env python3

from typing import Any, Callable, Sequence
import jax
import jax.numpy as jnp

from importlib import reload
import sys

from typing import List, Optional, cast

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import PhysicalField
from jax_spectral_dns.equation import Equation
from jax_spectral_dns._typing import jsd_float, jnp_array, Vel_fn_type


class Heat_Eq(Equation):
    name: str = "heat equation"

    def __init__(self, domain: PhysicalDomain, *fields: PhysicalField, **params: Any):
        super().__init__(domain, *fields)
        u = self.get_initial_field("u")
        u.update_boundary_conditions()
        self.dt: jsd_float = params["dt"]
        self.number_of_steps: int = params["number_of_steps"]
        self.i: int = 0

    def get_field(self, name: str, index: int) -> "PhysicalField":
        out = cast(PhysicalField, super().get_field(name, index))
        return out

    def get_fields(self, name: str) -> list["PhysicalField"]:
        return cast(List[PhysicalField], super().get_fields(name))

    def get_initial_field(self, name: str) -> "PhysicalField":
        out = cast(PhysicalField, super().get_initial_field(name))
        return out

    def get_latest_field(self, name: str) -> "PhysicalField":
        out = cast(PhysicalField, super().get_latest_field(name))
        return out

    def perform_explicit_euler_step(self) -> None:
        dt = self.dt
        i = self.i
        u = self.get_latest_field("u")
        eq = u.laplacian()
        new_u = u + eq * dt
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        # self.fields["u"].append(new_u)
        self.fields["u"][-1] = new_u

    def perform_implicit_euler_step(self) -> None:
        # only for 1D!
        dt = self.dt
        i = self.i
        u = self.get_latest_field("u")
        assert isinstance(u, PhysicalField)
        D2 = self.get_domain().get_cheb_mat_2_homogeneous_dirichlet(0)
        I = jnp.eye(D2.shape[0])
        new_u = PhysicalField(
            u.get_domain(), jnp.linalg.inv(I - dt * D2) @ u.data, name=u.name
        )
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        # self.fields["u"].append(new_u)
        self.fields["u"][-1] = new_u
        out = self.fields["u"]

    def perform_time_step(self, _: Optional[Any] = None) -> None:
        self.perform_explicit_euler_step()
        self.i += 1
        # return self.perform_implicit_euler_step(dt, i)

    def solve(self) -> list[PhysicalField]:
        for i in jnp.arange(1, self.number_of_steps + 1):
            self.perform_time_step()
        out: list[PhysicalField] = self.get_fields("u")
        return out

    def plot(self) -> None:
        if self.get_domain().number_of_dimensions <= 2:
            u_0: PhysicalField = self.get_initial_field("u")
            u_fin: PhysicalField = self.get_latest_field("u")
            u_0.plot(u_fin)
        elif self.get_domain().number_of_dimensions == 3:
            u_0 = self.get_initial_field("u")
            u_fin = self.get_latest_field("u")
            for i in self.all_dimensions():
                u_0.plot_center(i, u_fin)


def solve_heat_eq_1D() -> None:
    # Nx = 100000
    Nx = 100
    dt = 5e-9
    Nt = 3000

    domain = PhysicalDomain.create((Nx,), (False,))

    u_fn: Vel_fn_type = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")
    heat_eq = Heat_Eq(domain, u, dt=dt, number_of_steps=Nt)

    # def before_time_step(heat_eq):
    #     print("step", heat_eq.time_step)
    # heat_eq.before_time_step_fn = before_time_step
    heat_eq.solve()
    heat_eq.plot()
    v_final = heat_eq.get_latest_field("u")
    e0 = u.energy()
    print(e0)
    print(v_final.energy())


def optimize_heat_eq_1D() -> None:
    Nx = 100

    def run(v0: jnp_array) -> jsd_float:
        domain = PhysicalDomain.create((Nx,), (False,))

        u = PhysicalField(domain, v0, name="u")
        e0 = u.energy()

        Nt = 3000
        dt = 5e-9
        heat_eq = Heat_Eq(domain, u, dt=dt, number_of_steps=Nt)

        heat_eq.solve()
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


def solve_heat_eq_2D() -> None:
    Nx = 24
    Ny = Nx
    Nt = 5000
    dt = 5e-5

    domain = PhysicalDomain.create((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")

    heat_eq = Heat_Eq(domain, u, dt=dt, number_of_steps=Nt)

    heat_eq.solve()
    heat_eq.plot()


def solve_heat_eq_3D() -> None:
    Nx = 100
    Ny = 100
    Nz = 100
    Nt = 3000
    dt = 5e-9

    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u = PhysicalField.FromFunc(domain, func=u_fn, name="u")
    heat_eq = Heat_Eq(domain, u, dt=dt, number_of_steps=Nt)

    # def before_time_step(heat_eq):
    #     print("step", heat_eq.time_step)
    # heat_eq.before_time_step_fn = before_time_step
    heat_eq.solve()
    heat_eq.plot()
    v_final = heat_eq.get_latest_field("u")
    e0 = u.energy()
    print(e0)
    print(v_final.energy())


def optimize_heat_eq_3D() -> None:
    Nx = 100
    Ny = Nx
    Nz = Nx

    def run(v0: jnp_array) -> jsd_float:
        domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))
        Nt = 3000
        dt = 5e-9

        u = PhysicalField(domain, v0, name="u")
        e0 = u.energy()

        heat_eq = Heat_Eq(domain, u, dt=dt, number_of_steps=Nt)

        heat_eq.solve()
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

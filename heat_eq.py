#!/usr/bin/env python3

import jax.numpy as jnp

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    pass
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    pass
from field import Field

try:
    reload(sys.modules["equation"])
except:
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
        self.fields["u"].append(new_u)
        return self.fields["u"]

    def perform_time_step(self, dt, i):
        return self.perform_explicit_euler_step(dt, i)

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


def solve_heat_eq_2D():
    Nx = 24
    Ny = Nx

    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u")

    heat_eq = Heat_Eq(domain, u)

    Nt = 5000
    dt = 5e-5
    heat_eq.solve(dt, Nt)
    heat_eq.plot()


def solve_heat_eq_3D():
    Nx = 24
    Ny = Nx
    Nz = Nx

    domain = Domain((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u")
    heat_eq = Heat_Eq(domain, u)

    Nt = 5000
    dt = 5e-5
    heat_eq.solve(dt, Nt)
    heat_eq.plot()

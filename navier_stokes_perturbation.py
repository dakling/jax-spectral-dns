#!/usr/bin/env python3

NoneType = type(None)
import jax
import jax.numpy as jnp
from functools import partial

# from importlib import reload
import sys

from navier_stokes import NavierStokesVelVort

# try:
#     reload(sys.modules["domain"])
# except:
#     if hasattr(sys, 'ps1'):
#         pass
from domain import PhysicalDomain

# try:
#     reload(sys.modules["field"])
# except:
#     if hasattr(sys, 'ps1'):
#         pass
from field import PhysicalField, VectorField, FourierField, FourierFieldSlice

# try:
#     reload(sys.modules["equation"])
# except:
#     if hasattr(sys, 'ps1'):
#         pass
from equation import Equation


# @partial(jax.jit, static_argnums=(0,))
def update_nonlinear_terms_high_performance_perturbation(
    domain, vel_hat_new, vel_base_hat, linearize=False
):
    vel_new = jnp.array(
        [
            # domain.no_hat(vel_hat_new.at[i].get())
            domain.no_hat(vel_hat_new[i,...])
            for i in domain.all_dimensions()
        ]
    )
    vort_new = domain.curl(vel_new)

    vel_new_sq = 0
    for j in domain.all_dimensions():
        vel_new_sq += vel_new[j,...] * vel_new[j,...]
    vel_new_sq_nabla = []
    for i in domain.all_dimensions():
        vel_new_sq_nabla.append(domain.diff(vel_new_sq, i))

    # hel_new_ = domain.cross_product(vel_new, vort_new)
    # conv_ns_new_ = -jnp.array(hel_new_) + 1 / 2 * jnp.array(vel_new_sq_nabla)
    hel_new_ = jnp.array(domain.cross_product(vel_new, vort_new)) - 1 / 2 * jnp.array(
        vel_new_sq_nabla
    )

    # a-term
    vel_base = jnp.array(
        [
            domain.no_hat(vel_base_hat.at[i].get())
            for i in jnp.arange(domain.number_of_dimensions)
        ]
    )
    vel_new_sq_a = 0
    for j in domain.all_dimensions():
        vel_new_sq_a += vel_new[j] * vel_base[j]
    vel_new_sq_nabla_a = []
    for i in domain.all_dimensions():
        vel_new_sq_nabla_a.append(domain.diff(vel_new_sq_a, i))
    hel_new_a = jnp.array(domain.cross_product(vel_base, vort_new)) - 1 / 2 * jnp.array(
        vel_new_sq_nabla_a
    )

    # b-term
    vort_base = domain.curl(vel_base)
    vel_new_sq_nabla_b = vel_new_sq_nabla_a
    hel_new_b = jnp.array(domain.cross_product(vel_new, vort_base)) - 1 / 2 * jnp.array(
        vel_new_sq_nabla_b
    )

    # hel_new = (0.0 if linearize else 1.0) * hel_new_ + hel_new_a + hel_new_b
    hel_new =  jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * hel_new_ + hel_new_a + hel_new_b
    conv_ns_new = -hel_new

    h_v_new = (
        -domain.diff(domain.diff(hel_new[0], 0) + domain.diff(hel_new[2], 2), 1)
        + domain.diff(hel_new[1], 0, 2)
        + domain.diff(hel_new[1], 2, 2)
    )
    h_g_new = domain.diff(hel_new[0], 2) - domain.diff(hel_new[2], 0)

    h_v_hat_new = domain.field_hat(h_v_new)
    h_g_hat_new = domain.field_hat(h_g_new)
    vort_hat_new = [domain.field_hat(vort_new[i]) for i in domain.all_dimensions()]
    conv_ns_hat_new = [
        domain.field_hat(conv_ns_new[i]) for i in domain.all_dimensions()
    ]

    return (h_v_hat_new, h_g_hat_new, vort_hat_new, conv_ns_hat_new)
    # return (jnp.zeros_like(h_v_hat_new), jnp.zeros_like(h_g_hat_new), [jnp.zeros_like(vort_hat_new[i]) for i in range(3)], [jnp.zeros_like(conv_ns_hat_new[i]) for i in range(3)])


class NavierStokesVelVortPerturbation(NavierStokesVelVort):
    name = "Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a base flow."
    max_cfl = 0.7
    # max_dt = 1e10
    max_dt = 2e-2

    def __init__(self, velocity_field, **params):

        super().__init__(velocity_field, **params)

        velocity_x_base = PhysicalField.FromFunc(
            self.physical_domain,
            lambda X: self.u_max_over_u_tau * (1 - X[1] ** 2) + 0.0 * X[0] * X[2],
            name="velocity_x_base",
        )
        velocity_y_base = PhysicalField.FromFunc(
            self.physical_domain,
            lambda X: 0.0 * X[0] * X[1] * X[2],
            name="velocity_y_base",
        )
        velocity_z_base = PhysicalField.FromFunc(
            self.physical_domain,
            lambda X: 0.0 * X[0] * X[1] * X[2],
            name="velocity_z_base",
        )
        velocity_base_hat = VectorField(
            [velocity_x_base.hat(), velocity_y_base.hat(), velocity_z_base.hat()]
        )
        velocity_base_hat.set_name("velocity_base_hat")
        self.add_field("velocity_base_hat", velocity_base_hat)

        try:
            self.linearize = params["linearize"]
        except KeyError:
            self.linearize = False
        self.set_linearize(self.linearize)

    def update_flow_rate(self):
        self.flow_rate = 0.0
        self.dpdx = PhysicalField.FromFunc(
            self.physical_domain, lambda X: 0.0 * X[0] * X[1] * X[2]
        ).hat()
        self.dpdz = PhysicalField.FromFunc(
            self.physical_domain, lambda X: 0.0 * X[0] * X[1] * X[2]
        ).hat()

    def set_linearize(self, lin):
        self.linearize = lin
        velocity_base_hat = self.get_latest_field("velocity_base_hat")
        self.nonlinear_update_fn = (
            lambda dom, vel: update_nonlinear_terms_high_performance_perturbation(
                dom,
                vel,
                jnp.array(
                    [
                        velocity_base_hat[0].data,
                        velocity_base_hat[1].data,
                        velocity_base_hat[2].data,
                    ]
                ),
                linearize=self.linearize,
            ))

    def get_time_step(self):
        return self.max_dt
        if self.time_step % self.dt_update_frequency == 0:
            dX = self.physical_domain.grid[0][1:] - self.physical_domain.grid[0][:-1]
            dY = self.physical_domain.grid[1][1:] - self.physical_domain.grid[1][:-1]
            dZ = self.physical_domain.grid[2][1:] - self.physical_domain.grid[2][:-1]
            DX, DY, DZ = jnp.meshgrid(dX, dY, dZ, indexing="ij")
            vel = self.get_latest_field("velocity_hat").no_hat()
            vel_base = self.get_latest_field("velocity_base_hat").no_hat()
            U = vel[0][1:, 1:, 1:] + vel_base[0][1:, 1:, 1:]
            V = vel[1][1:, 1:, 1:] + vel_base[1][1:, 1:, 1:]
            W = vel[2][1:, 1:, 1:] + vel_base[2][1:, 1:, 1:]
            u_cfl = (abs(DX) / abs(U)).min().real
            v_cfl = (abs(DY) / abs(V)).min().real
            w_cfl = (abs(DZ) / abs(W)).min().real
            self.dt = min(self.max_dt, self.max_cfl * min([u_cfl, v_cfl, w_cfl]))
            assert self.dt > 1e-8, "Breaking due to small timestep, which indicates an issue with the calculation."
        return self.dt


def solve_navier_stokes_perturbation(
    Re=1.8e2,
    end_time=1e1,
    max_iter=1e8,
    Nx=6,
    Ny=40,
    Nz=None,
    perturbation_factor=0.1,
    scale_factors=(1.87, 1.0, 0.93),
):
    Ny = Ny
    Nz = Nz or Nx + 4

    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)

    vel_x_fn = lambda X: (
        0.1
        * perturbation_factor
        * (
            jnp.pi
            / 3
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(3 * X[0])
            * jnp.cos(4 * X[2])
        )
    )
    # add small perturbation in y and z to see if it decays
    vel_y_fn = (
        lambda X: 0.1
        * perturbation_factor
        * (
            jnp.pi
            / 3
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(3 * X[0])
            * jnp.cos(4 * X[2])
        )
    )
    vel_z_fn = (
        lambda X: 0.1
        * jnp.pi
        / 3
        * perturbation_factor
        * (jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(5 * X[0]) * jnp.cos(3 * X[2]))
    )
    vel_x = PhysicalField.FromFunc(domain, vel_x_fn, name="vel_x")
    vel_y = PhysicalField.FromFunc(domain, vel_y_fn, name="vel_y")
    vel_z = PhysicalField.FromFunc(domain, vel_z_fn, name="vel_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")

    nse = NavierStokesVelVortPerturbation.FromVelocityField(vel, Re)
    nse.end_time = end_time
    nse.max_iter = max_iter

    nse.after_time_step_fn = None

    return nse

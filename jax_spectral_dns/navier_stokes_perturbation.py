#!/usr/bin/env python
from __future__ import annotations

NoneType = type(None)
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.figure as figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any, Tuple, cast

# from importlib import reload
import sys

from jax_spectral_dns.navier_stokes import (
    NavierStokesVelVort,
    helicity_to_nonlinear_terms,
)
from jax_spectral_dns.domain import PhysicalDomain, FourierDomain
from jax_spectral_dns.field import (
    PhysicalField,
    VectorField,
    FourierField,
    FourierFieldSlice,
)
from jax_spectral_dns.equation import Equation, print_verb

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        jsd_float,
        jnp_array,
        np_jnp_array,
        Vel_fn_type,
    )


def update_nonlinear_terms_high_performance_perturbation_convection(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_base_hat: "jnp_array",
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    vel_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_base = jnp.array(
        [
            fourier_domain.field_no_hat(vel_base_hat.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
        ]
    )

    n = fourier_domain.number_of_dimensions
    # TODO can I make this more efficient?
    dvel_hat_i_dx_j = jnp.array(
        [[fourier_domain.diff(vel_hat_new[i], j) for j in range(n)] for i in range(n)]
    )
    dvel_base_i_dx_j = jnp.array(
        [[fourier_domain.diff(vel_base_hat[i], j) for j in range(n)] for i in range(n)]
    )

    vel_new_nabla_vel_new = jnp.sum(
        jnp.array(
            [
                [
                    vel_new[j] * fourier_domain.field_no_hat(dvel_hat_i_dx_j[i, j])
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )
    vel_new_nabla_vel_base = jnp.sum(
        jnp.array(
            [
                [
                    vel_new[j] * fourier_domain.field_no_hat(dvel_base_i_dx_j[i, j])
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )
    vel_base_nabla_vel_new = jnp.sum(
        jnp.array(
            [
                [
                    vel_base[j] * fourier_domain.field_no_hat(dvel_hat_i_dx_j[i, j])
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )

    # vel_new_nabla_vel_new = []
    # vel_new_nabla_vel_base = []
    # vel_base_nabla_vel_new = []
    # for i in physical_domain.all_dimensions():
    #     vel_new_nabla_vel_new_i = jnp.zeros_like(vel_new[0])
    #     vel_new_nabla_vel_base_i = jnp.zeros_like(vel_new[0])
    #     vel_base_nabla_vel_new_i = jnp.zeros_like(vel_new[0])
    #     for j in physical_domain.all_dimensions():
    #         # the nonlinear term
    #         vel_u_hat_i_diff_j = fourier_domain.diff(vel_hat_new[i], j)
    #         vel_new_nabla_vel_new_i += vel_new[j] * fourier_domain.field_no_hat(
    #             vel_u_hat_i_diff_j
    #         )
    #         # the a part
    #         vel_u_base_hat_i_diff_j = fourier_domain.diff(vel_base_hat[i], j)
    #         vel_new_nabla_vel_base_i += vel_new[j] * fourier_domain.field_no_hat(
    #             vel_u_base_hat_i_diff_j
    #         )
    #         # the b part
    #         vel_base_nabla_vel_new_i += vel_base[j] * fourier_domain.field_no_hat(
    #             vel_u_hat_i_diff_j
    #         )
    #     vel_new_nabla_vel_new.append(vel_new_nabla_vel_new_i)
    #     vel_new_nabla_vel_base.append(vel_new_nabla_vel_base_i)
    #     vel_base_nabla_vel_new.append(vel_base_nabla_vel_new_i)
    # vel_new_nabla_vel_new_ = jnp.array(vel_new_nabla_vel_new)
    # vel_new_nabla_vel_base_ = jnp.array(vel_new_nabla_vel_base)
    # vel_base_nabla_vel_new_ = jnp.array(vel_base_nabla_vel_new)
    hel_new = -(
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_new_nabla_vel_new
        + vel_base_nabla_vel_new
        + vel_new_nabla_vel_base
    )

    hel_new_hat = jnp.array(
        [
            physical_domain.field_hat(hel_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


def update_nonlinear_terms_high_performance_perturbation_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_base_hat: "jnp_array",
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    vel_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_base = jnp.array(
        [
            fourier_domain.field_no_hat(vel_base_hat.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
        ]
    )

    n = fourier_domain.number_of_dimensions
    # TODO can I make this more efficient?
    nabla_vel_new_vel_new = jnp.sum(
        jnp.array(
            [
                [
                    fourier_domain.diff(
                        physical_domain.field_hat(vel_new[i] * vel_new[j]), j
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )
    nabla_vel_base_vel_new = jnp.sum(
        jnp.array(
            [
                [
                    fourier_domain.diff(
                        physical_domain.field_hat(vel_base[i] * vel_new[j]), j
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )
    nabla_vel_new_vel_base = jnp.sum(
        jnp.array(
            [
                [
                    fourier_domain.diff(
                        physical_domain.field_hat(vel_new[i] * vel_base[j]), j
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
        ),
        axis=1,
    )

    # nabla_vel_new_vel_new = []
    # nabla_vel_base_vel_new = []
    # nabla_vel_new_vel_base = []
    # for i in physical_domain.all_dimensions():
    #     nabla_vel_new_vel_new_i = jnp.zeros_like(vel_hat_new[0])
    #     nabla_vel_base_vel_new_i = jnp.zeros_like(vel_hat_new[0])
    #     nabla_vel_new_vel_base_i = jnp.zeros_like(vel_hat_new[0])
    #     for j in physical_domain.all_dimensions():
    #         # the nonlinear term
    #         vel_u_i_u_j = vel_new[i] * vel_new[j]
    #         vel_u_i_u_j_hat = physical_domain.field_hat(vel_u_i_u_j)
    #         nabla_vel_new_vel_new_i += fourier_domain.diff(vel_u_i_u_j_hat, j)
    #         # the a part
    #         vel_u_i_u_base_j = vel_new[i] * vel_base[j]
    #         vel_u_i_u_base_j_hat = physical_domain.field_hat(vel_u_i_u_base_j)
    #         nabla_vel_base_vel_new_i += fourier_domain.diff(vel_u_i_u_base_j_hat, j)
    #         # the b part
    #         vel_u_base_i_u_j = vel_base[i] * vel_new[j]
    #         vel_u_base_i_u_j_hat = physical_domain.field_hat(vel_u_base_i_u_j)
    #         nabla_vel_new_vel_base_i += fourier_domain.diff(vel_u_base_i_u_j_hat, j)
    #     nabla_vel_new_vel_new.append(nabla_vel_new_vel_new_i)
    #     nabla_vel_base_vel_new.append(nabla_vel_base_vel_new_i)
    #     nabla_vel_new_vel_base.append(nabla_vel_new_vel_base_i)
    # nabla_vel_new_vel_new_ = jnp.array(nabla_vel_new_vel_new)
    # nabla_vel_base_vel_new_ = jnp.array(nabla_vel_base_vel_new)
    # nabla_vel_new_vel_base_ = jnp.array(nabla_vel_new_vel_base)
    hel_new_hat = -(
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * nabla_vel_new_vel_new
        + nabla_vel_base_vel_new
        + nabla_vel_new_vel_base
    )

    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


def update_nonlinear_terms_high_performance_perturbation_skew_symmetric(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_base_hat: "jnp_array",
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    out = tuple(
        [
            0.5
            * (
                update_nonlinear_terms_high_performance_perturbation_convection(
                    physical_domain,
                    fourier_domain,
                    vel_hat_new,
                    vel_base_hat,
                    linearize,
                )[i]
                + update_nonlinear_terms_high_performance_perturbation_diffusion(
                    physical_domain,
                    fourier_domain,
                    vel_hat_new,
                    vel_base_hat,
                    linearize,
                )[i]
            )
            for i in range(4)
        ]
    )
    return cast(Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"], out)


# @partial(jax.jit, static_argnums=(0, 1))
def update_nonlinear_terms_high_performance_perturbation_rotational(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_base_hat: "jnp_array",
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vort_hat_new = fourier_domain.curl(vel_hat_new)
    vel_new = jnp.array(
        [
            fourier_domain.filter_field_nonfourier_only(
                fourier_domain.field_no_hat(vel_hat_new[i])
            )
            # fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vort_new = jnp.array(
        [
            fourier_domain.filter_field_nonfourier_only(
                fourier_domain.field_no_hat(vort_hat_new[i])
            )
            # fourier_domain.field_no_hat(vort_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    vel_new_sq = jnp.zeros_like(vel_new[0])
    for j in physical_domain.all_dimensions():
        vel_new_sq += vel_new[j] * vel_new[j]
    vel_new_sq_hat = physical_domain.field_hat(vel_new_sq)
    vel_new_sq_hat_nabla = []
    for i in physical_domain.all_dimensions():
        vel_new_sq_hat_nabla.append(fourier_domain.diff(vel_new_sq_hat, i))

    vel_vort_new = physical_domain.cross_product(vel_new, vort_new)
    vel_vort_new_hat = jnp.array(
        [
            physical_domain.field_hat(vel_vort_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    hel_new_hat = vel_vort_new_hat - 1 / 2 * jnp.array(vel_new_sq_hat_nabla)

    # a-term
    vel_base = jnp.array(
        [
            fourier_domain.field_no_hat(vel_base_hat.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
        ]
    )
    vel_new_sq_a = jnp.zeros_like(vel_new[0, ...])
    for j in physical_domain.all_dimensions():
        vel_new_sq_a += vel_new[j] * vel_base[j]
    vel_new_sq_a_hat = physical_domain.field_hat(vel_new_sq_a)
    vel_new_sq_a_hat_nabla = []
    for i in physical_domain.all_dimensions():
        vel_new_sq_a_hat_nabla.append(fourier_domain.diff(vel_new_sq_a_hat, i))
    vel_vort_new_a = physical_domain.cross_product(vel_base, vort_new)
    vel_vort_new_a_hat = jnp.array(
        [
            physical_domain.field_hat(vel_vort_new_a[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    hel_new_a_hat = jnp.array(vel_vort_new_a_hat) - 1 / 2 * jnp.array(
        vel_new_sq_a_hat_nabla
    )

    # b-term
    vort_base_hat = fourier_domain.curl(vel_base_hat)
    vort_base = jnp.array(
        [
            fourier_domain.field_no_hat(vort_base_hat.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
        ]
    )
    vel_new_sq_b_hat_nabla = vel_new_sq_a_hat_nabla
    vel_vort_new_b = physical_domain.cross_product(vel_new, vort_base)
    vel_vort_new_b_hat = jnp.array(
        [
            physical_domain.field_hat(vel_vort_new_b[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    hel_new_b_hat = vel_vort_new_b_hat - 1 / 2 * jnp.array(vel_new_sq_b_hat_nabla)

    # hel_new = (0.0 if linearize else 1.0) * hel_new_ + hel_new_a + hel_new_b
    hel_new_hat = (
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * hel_new_hat
        + hel_new_a_hat
        + hel_new_b_hat
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


class NavierStokesVelVortPerturbation(NavierStokesVelVort):
    name = "Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a base flow."
    max_cfl = 0.5
    # max_dt = 1e10
    max_dt = 2e-2

    def __init__(self, velocity_field: VectorField[FourierField], **params: Any):

        super().__init__(velocity_field, **params)

        try:
            velocity_base_hat = params["velocity_base_hat"]
            if not params.get("non_verbose", False):
                print_verb("Using provided velocity base profile", verbosity_level=2)
        except KeyError:
            if not params.get("non_verbose", False):
                print_verb(
                    "Using default laminar velocity base profile", verbosity_level=2
                )
            velocity_x_base = PhysicalField.FromFunc(
                self.get_physical_domain(),
                lambda X: self.get_u_max_over_u_tau() * (1 - X[1] ** 2)
                + 0.0 * X[0] * X[2],
                name="velocity_x_base",
            )
            velocity_y_base = PhysicalField.FromFunc(
                self.get_physical_domain(),
                lambda X: 0.0 * X[0] * X[1] * X[2],
                name="velocity_y_base",
            )
            velocity_z_base = PhysicalField.FromFunc(
                self.get_physical_domain(),
                lambda X: 0.0 * X[0] * X[1] * X[2],
                name="velocity_z_base",
            )
            velocity_base_hat = VectorField(
                [velocity_x_base.hat(), velocity_y_base.hat(), velocity_z_base.hat()]
            )
            velocity_base_hat.set_name("velocity_base_hat")
        self.add_field("velocity_base_hat", velocity_base_hat)

        self.linearize: bool = params.get("linearize", False)
        self.set_linearize(self.linearize)

    def update_flow_rate(self) -> None:
        self.flow_rate = 0.0
        self.dpdx = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: 0.0 * X[0] * X[1] * X[2]
        ).hat()
        self.dpdz = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: 0.0 * X[0] * X[1] * X[2]
        ).hat()

    def set_linearize(self, lin: bool) -> None:
        self.linearize = lin
        velocity_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        self.nonlinear_update_fn = lambda vel, _: update_nonlinear_terms_high_performance_perturbation_rotational(
            # self.nonlinear_update_fn = lambda vel, _: update_nonlinear_terms_high_performance_perturbation_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            jnp.array(
                [
                    velocity_base_hat[0].data,
                    velocity_base_hat[1].data,
                    velocity_base_hat[2].data,
                ]
            ),
            linearize=self.linearize,
        )

    def get_cfl(self, i: int = -1) -> "jnp_array":
        dX = (
            self.get_physical_domain().grid[0][1:]
            - self.get_physical_domain().grid[0][:-1]
        )
        dY = (
            self.get_physical_domain().grid[1][1:]
            - self.get_physical_domain().grid[1][:-1]
        )
        dZ = (
            self.get_physical_domain().grid[2][1:]
            - self.get_physical_domain().grid[2][:-1]
        )
        DX, DY, DZ = jnp.meshgrid(dX, dY, dZ, indexing="ij")
        vel_hat: VectorField[FourierField] = self.get_field("velocity_hat", i)
        vel: VectorField[PhysicalField] = vel_hat.no_hat()
        vel_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        vel_base: VectorField[PhysicalField] = vel_base_hat.no_hat()
        U = vel[0][1:, 1:, 1:] + vel_base[0][1:, 1:, 1:]
        V = vel[1][1:, 1:, 1:] + vel_base[1][1:, 1:, 1:]
        W = vel[2][1:, 1:, 1:] + vel_base[2][1:, 1:, 1:]
        u_cfl = cast(float, (abs(DX) / abs(U)).min().real)
        v_cfl = cast(float, (abs(DY) / abs(V)).min().real)
        w_cfl = cast(float, (abs(DZ) / abs(W)).min().real)
        return self.get_dt() / jnp.array([u_cfl, v_cfl, w_cfl])


def solve_navier_stokes_perturbation(
    Re: float = 1.8e2,
    end_time: float = 1e1,
    Nx: int = 8,
    Ny: int = 40,
    Nz: int = 8,
    perturbation_factor: float = 0.1,
    scale_factors: Tuple[float, float, float] = (1.87, 1.0, 0.93),
    dt: float = 1e-2,
    u_max_over_u_tau: "jsd_float" = 1.0,
    aliasing: float = 1.0,
    dealias_nonperiodic: bool = False,
    rotated: bool = False,
) -> NavierStokesVelVortPerturbation:

    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
        dealias_nonperiodic=dealias_nonperiodic,
    )

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
    if not rotated:
        vel_x = PhysicalField.FromFunc(domain, vel_x_fn, name="vel_x")
        vel_y = PhysicalField.FromFunc(domain, vel_y_fn, name="vel_y")
        vel_z = PhysicalField.FromFunc(domain, vel_z_fn, name="vel_z")
    else:
        vel_z = PhysicalField.FromFunc(domain, vel_x_fn, name="vel_z")
        vel_y = PhysicalField.FromFunc(domain, vel_y_fn, name="vel_y")
        vel_x = PhysicalField.FromFunc(domain, vel_z_fn, name="vel_x")
        velocity_z_base = PhysicalField.FromFunc(
            domain,
            lambda X: u_max_over_u_tau * (1 - X[1] ** 2) + 0.0 * X[0] * X[2],
            name="velocity_z_base",
        )
        velocity_y_base = PhysicalField.FromFunc(
            domain,
            lambda X: 0.0 * X[0] * X[1] * X[2],
            name="velocity_y_base",
        )
        velocity_x_base = PhysicalField.FromFunc(
            domain,
            lambda X: 0.0 * X[0] * X[1] * X[2],
            name="velocity_x_base",
        )
        velocity_base_hat = VectorField(
            [velocity_x_base.hat(), velocity_y_base.hat(), velocity_z_base.hat()]
        )
        velocity_base_hat.set_name("velocity_base_hat")

    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")

    if not rotated:
        nse = NavierStokesVelVortPerturbation.FromVelocityField(
            vel, Re=Re, dt=dt, end_time=end_time, prepare_matrices=True
        )
    else:
        nse = NavierStokesVelVortPerturbation.FromVelocityField(
            vel,
            Re=Re,
            dt=dt,
            velocity_base_hat=velocity_base_hat,
            end_time=end_time,
            prepare_matrices=True,
        )

    nse.before_time_step_fn = None
    nse.after_time_step_fn = None

    def post_process(nse_: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse_.get_number_of_fields("velocity_hat")
        vel_hat = nse_.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[0].plot_center(1)
        vel[1].plot_center(1)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) == Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse_.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")

    nse.set_post_process_fn(post_process)

    return nse

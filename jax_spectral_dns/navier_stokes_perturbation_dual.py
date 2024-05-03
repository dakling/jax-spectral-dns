#!/usr/bin/env python3

from __future__ import annotations

NoneType = type(None)
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.figure as figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any, Tuple, cast, List

# from importlib import reload
import sys

from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
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
        np_complex_array,
    )


def update_nonlinear_terms_high_performance_perturbation_dual(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_base_hat: "jnp_array",
    vel_u_hat: "jnp_array",
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vort_hat_new = fourier_domain.curl(vel_hat_new)
    vort_u_hat = fourier_domain.curl(vel_u_hat - vel_base_hat)
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
    vel_u_new = jnp.array(
        [
            fourier_domain.filter_field_nonfourier_only(
                fourier_domain.field_no_hat(vel_u_hat[i] + vel_base_hat[i])
            )
            # fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vort_u_new = jnp.array(
        [
            fourier_domain.filter_field_nonfourier_only(
                fourier_domain.field_no_hat(vort_u_hat[i])
            )
            # fourier_domain.field_no_hat(vort_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    vel_vort_new = physical_domain.cross_product(vel_new, vort_u_new)
    vel_vort_new_hat = jnp.array(
        [
            physical_domain.field_hat(vel_vort_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    hel_new_hat = vel_vort_new_hat

    conv_ns_hat_new = -hel_new_hat

    h_v_hat_new = (
        -fourier_domain.diff(
            fourier_domain.diff(hel_new_hat[0], 0)
            + fourier_domain.diff(hel_new_hat[2], 2),
            1,
        )
        + fourier_domain.diff(hel_new_hat[1], 0, 2)
        + fourier_domain.diff(hel_new_hat[1], 2, 2)
    )
    h_g_hat_new = fourier_domain.diff(hel_new_hat[0], 2) - fourier_domain.diff(
        hel_new_hat[2], 0
    )

    return (
        h_v_hat_new,
        h_g_hat_new,
        jnp.array(vort_hat_new),
        jnp.array(conv_ns_hat_new),
    )


class NavierStokesVelVortPerturbationDual(NavierStokesVelVortPerturbation):
    name = "Dual Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a base flow."

    def __init__(
        self,
        velocity_field: VectorField[FourierField],
        velocity_field_u_history: "jnp_array",
        **params: Any,
    ):

        super().__init__(velocity_field, **params)
        # self.add_field_history("velocity_u_hat", velocity_field_u_history)
        self.velocity_field_u_history = velocity_field_u_history

    def set_linearize(self, lin: bool) -> None:
        self.linearize = lin
        velocity_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            velocity_base_hat.get_data(),
            self.velocity_field_u_history.at[-1 - t].get(),
            # self.velocity_field_u_history[-1 - t],
            linearize=self.linearize,
        )

    def get_Re_tau(self) -> "jsd_float":
        return -self.nse_fixed_parameters.Re_tau

    def prepare_assemble_rk_matrices(
        self,
        domain: FourierDomain,
        physical_domain: PhysicalDomain,
        Re_tau: "jsd_float",
        dt: "jsd_float",
    ) -> Tuple["np_complex_array", ...]:
        return super().prepare_assemble_rk_matrices(
            domain, physical_domain, -Re_tau, -dt
        )


def perform_step_navier_stokes_perturbation_dual(
    nse: NavierStokesVelVortPerturbation,
) -> Tuple[jsd_float, jnp_array]:

    domain = nse.get_physical_domain()

    nse.write_entire_output = True
    nse.write_intermediate_output = False

    # nse.set_post_process_fn(post_process)
    velocity_u_hat_history_, _ = nse.solve_scan()
    velocity_u_hat_history = cast("jnp_array", velocity_u_hat_history_)
    u_hat_final = nse.get_latest_field("velocity_hat")
    u_hat_final.set_name("velocity_hat")

    u_hat_initial = nse.get_initial_field("velocity_hat")

    gain = u_hat_final.no_hat().energy() / u_hat_initial.no_hat().energy()

    Re = nse.get_Re_tau()
    dt = nse.get_dt()
    # TODO add constructor taking only nse as input argument
    nse_dual = NavierStokesVelVortPerturbationDual(
        u_hat_final, velocity_u_hat_history, Re=Re, dt=dt
    )

    nse_dual.end_time = nse.end_time

    nse_dual.before_time_step_fn = None
    nse_dual.after_time_step_fn = None
    nse_dual.write_entire_output = False
    nse_dual.write_intermediate_output = False

    nse_dual.solve_scan()

    vel_v_0_hat = nse_dual.get_latest_field("velocity_hat")
    lam = 1

    return (gain, lam * u_hat_initial.get_data() - vel_v_0_hat.get_data())

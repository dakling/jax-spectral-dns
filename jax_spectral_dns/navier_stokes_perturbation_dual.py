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
from typing_extensions import Self

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
    vel_hat_new: "jnp_array",  # v
    vel_base_hat: "jnp_array",  # U
    vel_u_hat: "jnp_array",  # u
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vort_u_hat = fourier_domain.curl(vel_u_hat + vel_base_hat)
    vel_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vort_u_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_u_hat[i])
            # )
            fourier_domain.field_no_hat(vort_u_hat[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    vort_hat_new = fourier_domain.curl(vel_hat_new)

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
        h_v_hat_new * 1,  # TODO
        h_g_hat_new * 1,
        jnp.array(vort_hat_new),
        jnp.array(conv_ns_hat_new) * 1,
    )


class NavierStokesVelVortPerturbationDual(NavierStokesVelVortPerturbation):
    name = "Dual Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a base flow."

    def __init__(
        self,
        velocity_field: VectorField[FourierField],
        velocity_field_u_history: "jnp_array",
        **params: Any,
    ):

        self.epsilon = params.get("epsilon", 1e-5)

        super().__init__(velocity_field, **params)
        # self.add_field_history("velocity_u_hat", velocity_field_u_history)
        self.velocity_field_u_history = velocity_field_u_history

    def set_linearize(self, lin: bool) -> None:
        self.linearize = lin
        velocity_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        self.nonlinear_update_fn = (
            lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual(
                self.get_physical_domain(),
                self.get_domain(),
                vel,
                velocity_base_hat.get_data(),
                self.velocity_field_u_history.at[-1 - t].get(),
                linearize=self.linearize,
            )
        )

    @classmethod
    def FromNavierStokesVelVortPerturbation(
        cls, nse: NavierStokesVelVortPerturbation
    ) -> Self:

        nse.write_entire_output = True
        nse.write_intermediate_output = False

        velocity_u_hat_history_, _ = nse.solve_scan()
        velocity_u_hat_history = cast("jnp_array", velocity_u_hat_history_)
        u_hat_final = nse.get_latest_field("velocity_hat")
        u_hat_final.set_name("velocity_hat")

        Re_tau = nse.get_Re_tau()
        dt = nse.get_dt()
        end_time = nse.end_time

        v_hat_initial = -1 * u_hat_final
        v_hat_initial.set_name("velocity_hat")
        nse_dual = cls(
            v_hat_initial,
            velocity_u_hat_history,
            Re_tau=-Re_tau,
            dt=-dt,
            end_time=-end_time,
        )
        nse_dual.set_linearize(nse.linearize)
        return nse_dual

    # def get_Re_tau(self) -> "jsd_float":
    #     return -self.nse_fixed_parameters.Re_tau

    # def get_dt(self) -> "jsd_float":
    #     # return self.fixed_parameters.dt
    #     return -self.fixed_parameters.dt

    # def prepare_assemble_rk_matrices(
    #     self,
    #     domain: FourierDomain,
    #     physical_domain: PhysicalDomain,
    #     Re_tau: "jsd_float",
    #     dt: "jsd_float",
    # ) -> Tuple["np_complex_array", ...]:
    #     return super().prepare_assemble_rk_matrices(
    #         domain,
    #         physical_domain,
    #         -Re_tau,
    #         # dt,  # TODO
    #         -dt,  # TODO
    #     )


def perform_step_navier_stokes_perturbation_dual(
    nse: NavierStokesVelVortPerturbation, eps: float
) -> Tuple[jsd_float, "jnp_array"]:

    # nse.write_entire_output = True
    # nse.write_intermediate_output = False

    # # nse.set_post_process_fn(post_process)
    # velocity_u_hat_history_, _ = nse.solve_scan()
    # velocity_u_hat_history = cast("jnp_array", velocity_u_hat_history_)
    # u_hat_final = nse.get_latest_field("velocity_hat")
    # u_hat_final.set_name("velocity_hat")

    # u_hat_initial = nse.get_initial_field("velocity_hat")

    # gain = u_hat_final.no_hat().energy() / u_hat_initial.no_hat().energy()

    # Re_tau = nse.get_Re_tau()
    # dt = nse.get_dt()

    # v_hat_initial = -1 * u_hat_final
    # # v_hat_initial = -1 * u_hat_initial # for testing only
    # v_hat_initial.set_name("velocity_hat")
    # # TODO add constructor taking only nse as input argument
    # nse_dual = NavierStokesVelVortPerturbationDual(
    #     v_hat_initial, velocity_u_hat_history, Re_tau=-Re_tau, dt=-dt, end_time=-nse.end_time
    # )

    # nse_dual.end_time = nse.end_time
    # nse_dual.end_time = -nse.end_time

    nse_dual = NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
        nse
    )

    u_hat_initial = nse.get_initial_field("velocity_hat")
    u_hat_final = nse.get_latest_field("velocity_hat")
    gain = u_hat_final.no_hat().energy() / u_hat_initial.no_hat().energy()

    nse_dual.epsilon = eps
    nse_dual.before_time_step_fn = None
    nse_dual.after_time_step_fn = None
    nse_dual.write_entire_output = False
    nse_dual.write_intermediate_output = False
    nse_dual.activate_jit()

    nse_dual.solve()

    e0 = 1.0
    vel_v_0_hat = nse_dual.get_latest_field("velocity_hat")
    lam = 0.0  # TODO

    def get_new_energy_0(l: float) -> float:
        return cast(
            float,
            (
                (1 + nse_dual.epsilon * l) * u_hat_initial
                - nse_dual.epsilon * vel_v_0_hat
            )
            .no_hat()
            .energy(),
        )

    print_verb("optimising lambda...")
    i = 0
    while abs(get_new_energy_0(lam) - e0) > 1e-20 and i < 1000:
        lam = lam - (get_new_energy_0(lam) - e0) / jax.grad(get_new_energy_0)(lam)
        i += 1
    print_verb("optimising lambda done in", i, "iterations, lambda:", lam)
    print_verb("energy:", get_new_energy_0(lam))

    # lam = - vel_v_0_hat.max() / u_hat_initial.max()

    return (gain, lam * u_hat_initial.get_data() - vel_v_0_hat.get_data())

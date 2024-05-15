#!/usr/bin/env python3

from __future__ import annotations

NoneType = type(None)
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.figure as figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast, List
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

    vort_u_hat = fourier_domain.curl(
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_u_hat + vel_base_hat
    )
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
        jnp.array(vort_hat_new) * 1,
        jnp.array(conv_ns_hat_new) * 1,
    )


class NavierStokesVelVortPerturbationDual(NavierStokesVelVortPerturbation):
    name = "Dual Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a base flow."

    def __init__(
        self,
        velocity_field: Optional[VectorField[FourierField]],
        forward_equation: "NavierStokesVelVortPerturbation",
        **params: Any,
    ):

        self.epsilon = params.get("epsilon", 1e-5)

        if velocity_field is None:
            velocity_field_ = forward_equation.get_latest_field("velocity_hat") * 0.0
        else:
            velocity_field_ = velocity_field
        velocity_field_.set_name("velocity_hat")
        super().__init__(velocity_field_, **params)
        self.velocity_field_u_history: Optional["jnp_array"] = None
        self.forward_equation = forward_equation

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
            self.velocity_field_u_history.at[-1 - t].get(),  # type: ignore[union-attr]
            linearize=self.linearize,
        )

    @classmethod
    def FromNavierStokesVelVortPerturbation(
        cls, nse: NavierStokesVelVortPerturbation
    ) -> Self:

        nse.write_entire_output = True
        nse.write_intermediate_output = False

        Re_tau = nse.get_Re_tau()
        dt = nse.get_dt()
        end_time = nse.end_time

        nse_dual = cls(
            None,
            nse,
            Re_tau=-Re_tau,
            dt=-dt,
            end_time=-end_time,
        )
        nse_dual.set_linearize(nse.linearize)
        return nse_dual

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
        assert self.velocity_field_u_history is not None
        vel_u_hat: VectorField[FourierField] = VectorField.FromData(
            FourierField,
            self.get_physical_domain(),
            self.velocity_field_u_history[i, ...],
        )
        vel_u: VectorField[PhysicalField] = vel_u_hat.no_hat()
        vel_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        vel_base: VectorField[PhysicalField] = vel_base_hat.no_hat()
        U = vel_u[0][1:, 1:, 1:] + vel_base[0][1:, 1:, 1:]
        V = vel_u[1][1:, 1:, 1:] + vel_base[1][1:, 1:, 1:]
        W = vel_u[2][1:, 1:, 1:] + vel_base[2][1:, 1:, 1:]
        u_cfl = cast(float, (abs(DX) / abs(U)).min().real)
        v_cfl = cast(float, (abs(DY) / abs(V)).min().real)
        w_cfl = cast(float, (abs(DZ) / abs(W)).min().real)
        return self.get_dt() / jnp.array([u_cfl, v_cfl, w_cfl])

    # def get_Re_tau(self) -> "jsd_float":
    #     return -abs(self.nse_fixed_parameters.Re_tau)

    # def get_dt(self) -> "jsd_float":
    #     # return self.fixed_parameters.dt
    #     return -abs(self.fixed_parameters.dt)

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
    #         -abs(Re_tau),
    #         # dt,  # TODO
    #         -abs(dt),  # TODO
    #     )

    def is_forward_calculation_done(self) -> bool:
        return (
            self.forward_equation.get_number_of_fields("velocity_hat") > 1
            and type(self.velocity_field_u_history) is not NoneType
        )

    def is_backward_calculation_done(self) -> bool:
        return self.get_number_of_fields("velocity_hat") > 1

    def run_forward_calculation(self) -> None:
        nse = self.forward_equation
        nse.write_entire_output = True
        nse.write_intermediate_output = False
        if not self.is_forward_calculation_done():
            velocity_u_hat_history_, _ = nse.solve_scan()
            self.velocity_field_u_history = cast("jnp_array", velocity_u_hat_history_)
        self.set_field("velocity_hat", 0, -1 * nse.get_latest_field("velocity_hat"))
        self.forward_equation = nse  # not sure if this is necessary

    def run_backward_calculation(self) -> None:
        if not self.is_backward_calculation_done():
            self.run_forward_calculation()
            self.before_time_step_fn = None
            self.after_time_step_fn = None
            self.write_entire_output = False
            self.write_intermediate_output = False
            self.activate_jit()
            self.solve()

    def get_gain(self) -> float:
        self.run_forward_calculation()
        u_0 = self.forward_equation.get_initial_field("velocity_hat").no_hat()
        u_T = self.forward_equation.get_latest_field("velocity_hat").no_hat()
        return u_T.energy() / u_0.energy()

    def get_grad(self) -> "jnp_array":
        self.run_backward_calculation()
        u_hat_0 = self.forward_equation.get_initial_field("velocity_hat")
        v_hat_0 = self.get_latest_field("velocity_hat")
        gain = self.get_gain()
        e_0 = u_hat_0.no_hat().energy()
        return (gain * u_hat_0.get_data() - v_hat_0.get_data()) / e_0

    def get_projected_grad(self, step_size: float) -> "jnp_array":
        self.run_backward_calculation()
        u_hat_0 = self.forward_equation.get_initial_field("velocity_hat")
        v_hat_0 = self.get_latest_field("velocity_hat")
        e_0 = u_hat_0.no_hat().energy()
        lam = 0.0

        def get_new_energy_0(l: float) -> float:
            return (
                ((1 + step_size * l) * u_hat_0 - step_size * v_hat_0).no_hat().energy()
            )

        print_verb("optimising lambda...")
        i = 0
        while abs(get_new_energy_0(lam) - e_0) / e_0 > 1e-20 and i < 1000:
            lam += -(get_new_energy_0(lam) - e_0) / jax.grad(get_new_energy_0)(lam)
            i += 1
        print_verb("optimising lambda done in", i, "iterations, lambda:", lam)
        print_verb("energy:", get_new_energy_0(lam))

        return lam * u_hat_0.get_data() - v_hat_0.get_data()


def perform_step_navier_stokes_perturbation_dual(
    nse: NavierStokesVelVortPerturbation,
    step_size: float = 1e-2,
) -> Tuple[jsd_float, "jnp_array"]:

    nse_dual = NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
        nse
    )
    return (nse_dual.get_gain(), nse_dual.get_projected_grad(step_size))

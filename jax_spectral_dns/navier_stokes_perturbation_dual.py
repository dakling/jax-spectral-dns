#!/usr/bin/env python3

from __future__ import annotations
import gc

from jax_spectral_dns.navier_stokes import (
    get_div_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
    helicity_to_nonlinear_terms,
)

NoneType = type(None)
import os
import h5py  # type: ignore
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.figure as figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, cast, List, Union
from typing_extensions import Self
import time

try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    pass

# from importlib import reload
import sys

from jax_spectral_dns.fixed_parameters import NavierStokesVelVortFixedParameters
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
from jax_spectral_dns.navier_stokes import (
    get_nabla_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
    helicity_to_nonlinear_terms,
)
from jax_spectral_dns.domain import PhysicalDomain, FourierDomain
from jax_spectral_dns.field import (
    Field,
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


def get_helicity_perturbation_dual_convection(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_hat: "jnp_array",  # U
    vel_v_new: "jnp_array",  # v
    vel_u: "jnp_array",  # U
) -> "jnp_array":

    # the first term: - (U + u) \dot nabla v)
    vel_u_base_nabla_v = get_vel_1_nabla_vel_2(fourier_domain, vel_u, vel_v_hat_new)
    vel_u_base_nabla_v_hat = jnp.array(
        [
            physical_domain.field_hat(vel_u_base_nabla_v[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    # the second term: v \dot (\nabla (U + u))^T
    v_nabla_u_hat = []
    for i in physical_domain.all_dimensions():
        acc = jnp.zeros_like(vel_v_new[0])
        for j in physical_domain.all_dimensions():
            vel_u_hat_j_diff_i = fourier_domain.diff(vel_u_hat[j], i)
            acc += vel_v_new[j] * fourier_domain.field_no_hat(vel_u_hat_j_diff_i)
        acc_hat = jnp.array(physical_domain.field_hat(acc))
        v_nabla_u_hat.append(acc_hat)

    hel_new_hat = -vel_u_base_nabla_v_hat + jnp.array(v_nabla_u_hat)
    return hel_new_hat


def update_nonlinear_terms_high_performance_perturbation_dual_convection(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    re_ijj_hat: "jnp_array",
    linearise: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vel_u_hat = (
        jax.lax.cond(linearise, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
        + vel_u_base_hat
    )
    vel_v_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_u = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_u_hat[i])
            # )
            fourier_domain.field_no_hat(vel_u_hat[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    hel_new_hat = (
        get_helicity_perturbation_dual_convection(
            physical_domain,
            fourier_domain,
            vel_v_hat_new,  # v
            vel_u_hat,  # U
            vel_v_new,  # v
            vel_u,  # U
        )
        + re_ijj_hat
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


def get_helicity_perturbation_dual_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    linearise: bool = False,
) -> "jnp_array":
    vel_u_hat = (
        jax.lax.cond(linearise, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
        + vel_u_base_hat
    )
    vel_v_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_u = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_u_hat[i])
            # )
            fourier_domain.field_no_hat(vel_u_hat[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    # the first term: - nabla ((U + u) v)
    nabla_vel_new_vel_new_hat = get_nabla_vel_1_vel_2(
        physical_domain, fourier_domain, vel_v_new, vel_u, vel_v_hat_new
    )

    # the second term: v \dot (\nabla (U + u))^T
    v_nabla_u_hat = []
    for i in physical_domain.all_dimensions():
        acc = jnp.zeros_like(vel_v_new[0])
        for j in physical_domain.all_dimensions():
            vel_u_hat_j_diff_i = fourier_domain.diff(vel_u_hat[j], i)
            acc += vel_v_new[j] * fourier_domain.field_no_hat(vel_u_hat_j_diff_i)
        acc_hat = jnp.array(physical_domain.field_hat(acc))
        v_nabla_u_hat.append(acc_hat)

    hel_new_hat = -nabla_vel_new_vel_new_hat + jnp.array(v_nabla_u_hat)
    return hel_new_hat


def update_nonlinear_terms_high_performance_perturbation_dual_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    re_ijj_hat: "jnp_array",
    linearise: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    hel_new_hat = (
        get_helicity_perturbation_dual_diffusion(
            physical_domain,
            fourier_domain,
            vel_v_hat_new,  # v
            vel_u_base_hat,  # U
            vel_small_u_hat,  # u
            linearise,
        )
        + re_ijj_hat
    )

    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


def update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    re_ijj_hat: "jnp_array",
    linearise: bool = False,
    coupling_term: "bool" = True,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    vel_v_new = jnp.array(
        [
            fourier_domain.field_no_hat(vel_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_u_hat = (
        jax.lax.cond(linearise, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
        + vel_u_base_hat
    )
    vel_u = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_u_hat[i])
            # )
            fourier_domain.field_no_hat(vel_u_hat[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    div_vel_u_vel_v = get_div_vel_1_vel_2(fourier_domain, vel_u_hat, vel_v_new)
    div_vel_v_vel_u = get_div_vel_1_vel_2(fourier_domain, vel_v_hat_new, vel_u)
    div_vel_vel = 0.5 * (div_vel_u_vel_v + div_vel_v_vel_u)
    div_vel_vel_hat = jnp.array(
        [
            physical_domain.field_hat(div_vel_vel[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    hel_new_hat = (
        get_helicity_perturbation_dual_convection(
            physical_domain,
            fourier_domain,
            vel_v_hat_new,
            vel_u_hat,
            vel_v_new,
            vel_u,
        )
        + 0.5 * div_vel_vel_hat
        + re_ijj_hat
    )

    h_v_hat_new, h_g_hat_new, vort_hat_new, conv_ns_hat_new = (
        helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)
    )
    comb_corr_hy_hat = physical_domain.field_hat(
        physical_domain.diff(fourier_domain.field_no_hat(vel_u_base_hat[0]), 1)
        * vel_v_new[0]
    )
    comb_corr_hat = fourier_domain.diff(comb_corr_hy_hat, 0, 2) + fourier_domain.diff(
        comb_corr_hy_hat, 2, 2
    )
    h_v_hat_new_comb = jax.lax.cond(
        coupling_term, lambda: h_v_hat_new, lambda: h_v_hat_new + comb_corr_hat
    )

    conv_ns_hat_new_comb = jax.lax.cond(
        # coupling_term,
        True,  # TODO
        lambda: conv_ns_hat_new,
        lambda: conv_ns_hat_new
        - jnp.stack(
            [
                jnp.zeros_like(comb_corr_hy_hat),
                comb_corr_hy_hat,
                jnp.zeros_like(comb_corr_hy_hat),
            ]
        ),
    )  # probably irrelevant

    return (h_v_hat_new_comb, h_g_hat_new, vort_hat_new, conv_ns_hat_new_comb)


def update_nonlinear_terms_high_performance_perturbation_dual_rotational(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    re_ijj_hat: "jnp_array",
    linearise: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vel_u_hat = (
        jax.lax.cond(linearise, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
        + vel_u_base_hat
    )
    vel_v_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_u = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_u_hat[i])
            # )
            fourier_domain.field_no_hat(vel_u_hat[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    vort_v_hat_new = fourier_domain.curl(vel_v_hat_new)
    vort_v_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vort_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    # the first term: (U + u) \times (\nabla \times v)
    vel_u_vort_v_new = physical_domain.cross_product(vel_u, vort_v_new)
    vel_u_vort_v_new_hat = jnp.array(
        [
            physical_domain.field_hat(vel_u_vort_v_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    # the second term: \nabla ((U + u) \dot v)
    vel_uv_sq = jnp.zeros_like(vel_v_new[0])
    for j in physical_domain.all_dimensions():
        vel_uv_sq += vel_u[j] * vel_v_new[j]
    vel_uv_sq_hat = physical_domain.field_hat(vel_uv_sq)
    vel_uv_sq_nabla_hat = []
    for i in physical_domain.all_dimensions():
        vel_uv_sq_nabla_hat.append(fourier_domain.diff(vel_uv_sq_hat, i))

    # the third term: 2 v \dot (\nabla (U + u))^T
    v_nabla_u_hat = []
    for i in physical_domain.all_dimensions():
        acc = jnp.zeros_like(vel_v_new[0])
        for j in physical_domain.all_dimensions():
            vel_u_hat_j_diff_i = fourier_domain.diff(vel_u_hat[j], i)
            acc += 2 * vel_v_new[j] * fourier_domain.field_no_hat(vel_u_hat_j_diff_i)
        acc_hat = jnp.array(physical_domain.field_hat(acc))
        v_nabla_u_hat.append(acc_hat)

    hel_new_hat = (
        vel_u_vort_v_new_hat - jnp.array(vel_uv_sq_nabla_hat) + jnp.array(v_nabla_u_hat)
    )

    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


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
        self.dPdx_history: Optional[List["jsd_float"]] = None
        self.forward_equation = forward_equation
        self.velocity_u_hat_0 = forward_equation.get_initial_field("velocity_hat")
        calculation_size = self.end_time / self.get_dt()
        for i in self.all_dimensions():
            calculation_size *= self.get_domain().number_of_cells(i)
        checkpointing_threshold = params.get("checkpointing_threshold", 1.4e8)
        self.checkpointing = params.get(
            "checkpointing", calculation_size > checkpointing_threshold
        )
        if self.checkpointing:
            print_verb("using checkpointing to reduce peak memory usage")
            self.current_velocity_field_u_history: Optional["jnp_array"] = None
            self.current_u_history_start_step = -1
            self.current_dPdx_history: Optional[List["jsd_float"]] = None
            self.dPdx_fwd: jsd_float = 0.0
        else:
            print_verb("not using checkpointing")
        self.forward_equation.activate_jit()
        self.linearise_switch: Optional[float] = params.get("linearise_switch")

        linearise: bool = params.get("linearise", False)
        if self.linearise_switch is not None:
            assert (
                params.get("linearise") is None
            ), "need to either pass linearise or linearise_switch."
            lin_switch = self.linearise_switch
            number_of_time_steps = len(jnp.arange(0, self.end_time, self.get_dt()))
            self.linearise: Callable[[int], bool] = (
                lambda t: t + 1 >= number_of_time_steps * (1.0 - lin_switch)
            )
        else:
            self.linearise = lambda _: linearise

        coupling_term: bool = params.get("coupling_term", True)

        self.coupling_term_switch: Optional[float] = params.get("coupling_term_switch")

        if self.coupling_term_switch is not None:
            assert (
                params.get("coupling_term") is None
            ), "need to either pass coupling_term or coupling_term_switch."
            coupling_term_switch = self.coupling_term_switch
            number_of_time_steps = len(jnp.arange(0, self.end_time, self.get_dt()))
            self.coupling_term: Callable[[int], bool] = lambda t: not (
                t + 1 >= number_of_time_steps * coupling_term_switch
            )
        else:
            self.coupling_term = lambda _: coupling_term
        self.set_linearise()
        self.optimisation_modes = Enum(  # type: ignore[misc]
            "optimisation_modes", ["gain", "dissipation"]
        )
        self.optimisation_mode = self.optimisation_modes[
            params.get("optimisation_mode", "gain")
        ]
        print_verb("optimising for", self.get_objective_fun_name())
        self.write_trajectory = False

    def get_fixed_params(self) -> "NavierStokesVelVortFixedParameters":
        return self.forward_equation.nse_fixed_parameters

    def assemble_matrices(
        self, **params: Any
    ) -> Tuple[Optional["np_complex_array"], ...]:
        dt = self.get_dt()
        calculation_size = self.end_time / dt
        for i in self.all_dimensions():
            calculation_size *= self.get_domain().number_of_cells(i)
        prepare_matrices_threshold = params.get("prepare_matrices_threshold", 1e12)
        self.prepare_matrices = params.get(
            "prepare_matrices", calculation_size < prepare_matrices_threshold
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def set_linearise(self) -> None:
        # self.linearise = lin
        velocity_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        try:
            re_ijj_hat = self.get_latest_field("reynolds_stress_ijj_hat").get_data()
        except KeyError:
            re_ijj_hat = VectorField(
                [FourierField.Zeros(self.get_physical_domain()) for _ in range(3)]
            ).get_data()
        # self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual_rotational(
        if self.constant_mass_flux:
            print_verb("enforcing constant mass flux")
            self.flow_rate = 0.0
            self.dPdx = 0.0
            current_flow_rate = self.get_flow_rate()
            print_verb("mass flux before correction:", current_flow_rate)
            v0_corr = self.update_velocity_field(self.get_initial_field("velocity_hat"))
            self.set_initial_field("velocity_hat", v0_corr)
            print_verb("correcting initial condition to have zero mass flux")
            print_verb("mass flux after correction:", self.get_flow_rate())
        else:
            print_verb("enforcing constant pressure gradient")
            self.flow_rate = self.get_flow_rate()
            # if not self.linearise:
            #     self.dPdx = -(-1.0 + 0.0)
            #     self.source_x_00 = (
            #         -1
            #         / self.get_Re_tau()
            #         * velocity_base_hat[0].laplacian().get_data()[0, :, 0]
            #     )
            # else:
            #     self.dPdx = 0.0
            #     self.source_x_00 = None
            self.dPdx = 0.0
            self.source_x_00 = None
        self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            velocity_base_hat.get_data(),
            self.get_velocity_u_hat(t),
            re_ijj_hat,
            linearise=self.linearise(t),
            coupling_term=self.coupling_term(t),
        )

    @classmethod
    def FromNavierStokesVelVortPerturbation(
        cls, nse: NavierStokesVelVortPerturbation, **params: Any
    ) -> Self:

        nse.write_entire_output = True
        nse.write_intermediate_output = False

        Re_tau = nse.get_Re_tau()
        dt = nse.get_dt()
        end_time = nse.end_time

        stripped_params = {
            x: params[x]
            for x in params
            if x
            not in [
                "Re_tau",
                "dt",
                "end_time",
                "velocity_base_hat",
                "constant_mass_flux",
                "linearise",
                "linearise_switch",
                "coupling_term",
                "coupling_term_switch",
            ]
        }

        nse_dual = cls(
            None,
            nse,
            Re_tau=-Re_tau,
            dt=-dt,
            end_time=-end_time,
            velocity_base_hat=nse.get_latest_field("velocity_base_hat"),
            # reynolds_stress_ijj_hat=reynolds_stress_ijj_hat, # see farano 2017
            constant_mass_flux=nse.constant_mass_flux,
            linearise=nse.linearise(0) if nse.linearise_switch is None else None,
            linearise_switch=nse.linearise_switch,
            coupling_term=(
                nse.coupling_term(0) if nse.coupling_term_switch is None else None
            ),
            coupling_term_switch=nse.coupling_term_switch,
            **stripped_params,
        )
        nse_dual.set_linearise()
        return nse_dual

    def get_velocity_u_hat(self, timestep: int) -> "jnp_array":
        if self.checkpointing:
            assert self.current_velocity_field_u_history is not None
            return self.current_velocity_field_u_history[
                -1
                - (
                    timestep
                    - self.current_u_history_start_step * self.number_of_inner_steps
                )
            ]
        else:
            return self.velocity_field_u_history.at[-1 - timestep].get()  # type: ignore[union-attr]

    def get_dPdx(self, timestep: int) -> "jsd_float":
        if not self.constant_mass_flux:
            return 0.0
        if self.checkpointing:
            assert self.current_dPdx_history is not None
            ts = (
                timestep
                - self.current_u_history_start_step * self.number_of_inner_steps
            )
            return jax.lax.cond(
                ts >= len(self.current_dPdx_history),
                lambda: self.dPdx_fwd,
                lambda: self.current_dPdx_history[-1 - ts],
            )
            # return self.current_dPdx_history[
            #     -1
            #     - (
            #         timestep
            #         - self.current_u_history_start_step * self.number_of_inner_steps
            #     )
            # ]
        else:
            assert self.dPdx_history is not None
            return jax.lax.cond(
                timestep >= len(self.dPdx_history),
                lambda: 0.0,
                lambda: self.dPdx_history[-1 - timestep],
            )
            # return self.dPdx_history[-1 - timestep]

    def update_with_nse(self) -> None:
        self.forward_equation.write_entire_output = True
        self.forward_equation.write_intermediate_output = False
        self.clear_field("velocity_hat")
        self.dPdx = 0.0
        self.forward_equation.dPdx = 0.0
        self.velocity_field_u_history = None
        self.dPdx_history = None

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

    # TODO with or without this?
    # def enforce_constant_mass_flux(
    #     self, vel_new_hat_field: "jnp_array", _: "jsd_float", time_step: int
    # ) -> Tuple["jnp_array", "jsd_float"]:

    #     if self.constant_mass_flux:
    #         dPdx = -self.get_dPdx(time_step + 2)
    #         vel_new_hat_field = self.update_velocity_field_data(vel_new_hat_field)
    #     else:
    #         dPdx = 0.0

    #     return vel_new_hat_field, dPdx

    def solve_scan(
        self,
    ) -> Tuple[Union["jnp_array", VectorField[FourierField]], List["jsd_float"], int]:
        jax.clear_caches()  # type: ignore
        gc.collect()
        if not self.checkpointing:
            return super().solve_scan()
        else:
            nse = self.forward_equation
            self.number_of_time_steps = nse.number_of_time_steps
            self.number_of_outer_steps = nse.number_of_outer_steps
            self.number_of_inner_steps = nse.number_of_inner_steps

            # def get_inner_step_fn(
            #     current_velocity_field_u_history: "jnp_array",
            #     current_dPdx_history: List["jsd_float"],
            # ) -> Callable[
            #     [Tuple["jnp_array", "jsd_float", int], Any],
            #     Tuple[Tuple["jnp_array", "jsd_float", int], None],
            # ]:
            def inner_step_fn(
                u0: Tuple["jnp_array", "jsd_float", int], _: Any
            ) -> Tuple[Tuple["jnp_array", "jsd_float", int], None]:
                u0_, _, time_step = u0
                assert self.current_dPdx_history is not None
                dPdx = -self.get_dPdx(time_step + 1)
                out = self.perform_time_step(u0_, cast(float, dPdx), time_step)

                return ((out[0], out[1], time_step + 1), None)

                # return inner_step_fn

            def step_fn(u0: Tuple["jnp_array", "jsd_float", int], _: Any) -> Tuple[
                Tuple["jnp_array", "jsd_float", int],
                Tuple["jnp_array", "jsd_float", int],
            ]:
                jax.clear_caches()  # type: ignore
                gc.collect()
                timestep = u0[2]
                outer_start_step = timestep // self.number_of_inner_steps
                self.current_u_history_start_step = outer_start_step
                current_velocity_field_u_history, dPdx_history = (
                    self.run_forward_calculation_subrange(outer_start_step)
                )
                self.current_velocity_field_u_history = current_velocity_field_u_history
                self.current_dPdx_history = dPdx_history
                # inner_step_fn = get_inner_step_fn(
                #     current_velocity_field_u_history, dPdx_history
                # )
                out, _ = jax.lax.scan(
                    jax.checkpoint(inner_step_fn),  # type: ignore[attr-defined]
                    u0,
                    xs=None,
                    length=self.number_of_inner_steps,
                    # unroll=True,
                    # inner_step_fn, u0, xs=None, length=number_of_inner_steps
                )
                self.current_velocity_field_u_history = None
                self.current_dPdx_history = None
                return out, out

            u0 = self.get_initial_field("velocity_hat").get_data()
            if type(self.dPdx_history) is NoneType:
                self.dPdx_history = self.current_dPdx_history
            assert self.dPdx_history is not None

            if self.constant_mass_flux:
                dPdx = -self.dPdx_history[-1]
            else:
                dPdx = 0.0
            ts = jnp.arange(0, self.end_time, self.get_dt())

            if self.write_intermediate_output and not self.write_entire_output:
                u_final, trajectory = jax.lax.scan(
                    step_fn,
                    (u0, cast("jsd_float", dPdx), 0),
                    xs=None,
                    length=self.number_of_outer_steps,
                )
                for u in trajectory[0]:
                    velocity = VectorField(
                        [
                            FourierField(
                                self.get_physical_domain(),
                                u[i],
                                name="velocity_hat_" + "xyz"[i],
                            )
                            for i in self.all_dimensions()
                        ]
                    )
                    self.append_field("velocity_hat", velocity, in_place=False)
                # for i in range(self.get_number_of_fields("velocity_hat")):
                #     cfl_s = self.get_cfl(i)
                #     print_verb("i: ", i, "cfl:", cfl_s)
                cfl_final = self.get_cfl()
                print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
                return (trajectory[0], [u_final[1]], len(ts))
            elif self.write_entire_output:
                u_final, trajectory = jax.lax.scan(
                    step_fn,
                    (u0, cast("jsd_float", dPdx), 0),
                    xs=None,
                    length=self.number_of_outer_steps,
                )
                velocity_final = VectorField(
                    [
                        FourierField(
                            self.get_physical_domain(),
                            trajectory[0][-1][i],
                            name="velocity_hat_" + "xyz"[i],
                        )
                        for i in self.all_dimensions()
                    ]
                )
                self.append_field("velocity_hat", velocity_final, in_place=False)
                cfl_final = self.get_cfl()
                print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
                return (trajectory[0], [u_final[1]], len(ts))
            else:
                u_final, _ = jax.lax.scan(
                    step_fn,
                    (u0, cast("jsd_float", dPdx), 0),
                    xs=None,
                    length=self.number_of_outer_steps,
                )
                velocity_final = VectorField(
                    [
                        FourierField(
                            self.get_physical_domain(),
                            u_final[0][i],
                            name="velocity_hat_" + "xyz"[i],
                        )
                        for i in self.all_dimensions()
                    ]
                )
                self.append_field("velocity_hat", velocity_final, in_place=False)
                cfl_final = self.get_cfl()
                print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
                return (velocity_final, [u_final[1]], len(ts))

    def is_forward_calculation_done(self) -> bool:
        return (
            self.forward_equation.get_number_of_fields("velocity_hat") > 1
            and type(self.velocity_field_u_history) is not NoneType
        )

    def is_backward_calculation_done(self) -> bool:
        return self.get_number_of_fields("velocity_hat") > 1

    def run_forward_calculation(self) -> None:
        nse = self.forward_equation
        self.velocity_u_hat_0 = nse.get_initial_field("velocity_hat")
        nse.activate_jit()
        nse.write_intermediate_output = True
        nse.end_time = -1.0 * self.end_time
        if self.checkpointing:
            nse.write_entire_output = False
            self.current_velocity_field_u_history = None
        else:
            nse.write_entire_output = True
        if not self.is_forward_calculation_done():
            start_time = time.time()
            velocity_u_hat_history_, dPdx_history, _ = nse.solve_scan()
            iteration_duration = time.time() - start_time
            if (
                os.environ.get("JAX_SPECTRAL_DNS_FIELD_DIR") is not None
                and not self.checkpointing
                and self.write_trajectory
            ):
                print_verb("writing velocity trajectory to file...")

                with h5py.File(Field.field_dir + "/trajectory", "w") as f:
                    f.create_dataset(
                        "trajectory",
                        data=velocity_u_hat_history_,
                        compression="gzip",
                        compression_opts=9,
                    )
                print_verb("done writing velocity trajectory to file")
            else:
                print_verb("not writing velocity trajectory to file")
            try:
                print_verb(
                    "forward calculation took", format_timespan(iteration_duration)
                )
            except Exception:
                print_verb("forward calculation took", iteration_duration, "seconds")
            self.velocity_field_u_history = cast("jnp_array", velocity_u_hat_history_)
            self.dPdx_history = dPdx_history
            self.current_dPdx_history = dPdx_history
        if self.optimisation_mode == self.optimisation_modes.gain:
            self.set_initial_field(
                "velocity_hat", -1 * nse.get_latest_field("velocity_hat")
            )
        elif self.optimisation_mode == self.optimisation_modes.dissipation:
            self.set_initial_field(
                "velocity_hat", 0 * nse.get_latest_field("velocity_hat")
            )
        else:
            raise Exception(
                "unknown optimisation mode "
                + self.optimisation_mode
                + ". Valid choices are "
                + self.optimisation_modes
            )
        self.forward_equation = nse  # not sure if this is necessary
        jax.clear_caches()  # type: ignore
        gc.collect()

    def get_objective_fun(self) -> float:
        if self.optimisation_mode == self.optimisation_modes.gain:
            return self.get_gain()
        elif self.optimisation_mode == self.optimisation_modes.dissipation:
            self.gain = self.get_gain()  # TODO is this needed for gradient calculation?
            return self.get_dissipation_average()
        else:
            raise Exception(
                "unknown optimisation mode "
                + self.optimisation_mode
                + ". Valid choices are "
                + self.optimisation_modes
            )

    def get_objective_fun_name(self) -> "str":
        return cast("str", self.optimisation_mode.name)

    def get_source_term(self, timestep: int) -> Optional["jnp_array"]:
        if self.optimisation_mode == self.optimisation_modes.gain:
            return None
        elif self.optimisation_mode == self.optimisation_modes.dissipation:
            return (
                -1
                / jnp.abs(self.get_Re_tau() * self.end_time)
                * jnp.array(
                    [
                        self.get_domain().laplacian(
                            self.get_velocity_u_hat(timestep)[i]
                        )
                        for i in range(3)
                    ]
                )
            )
        else:
            raise Exception(
                "unknown optimisation mode "
                + self.optimisation_mode
                + ". Valid choices are "
                + self.optimisation_modes
            )

    def run_forward_calculation_subrange(
        self, outer_timestep: int
    ) -> Tuple["jnp_array", List["jsd_float"]]:
        nse = self.forward_equation
        nse.write_intermediate_output = True
        nse.write_entire_output = True
        nse.clear_field("velocity_hat")
        assert self.velocity_field_u_history is not None
        step = -1 - (outer_timestep + 1)
        init_field_data = self.velocity_field_u_history[step]
        init_field: VectorField[FourierField] = VectorField.FromData(
            FourierField,
            self.get_physical_domain(),
            init_field_data,
            name="velocity_hat",
        )
        assert self.dPdx_history is not None
        if self.constant_mass_flux:
            dPdx = jax.lax.cond(
                outer_timestep >= self.number_of_outer_steps - 1,
                lambda: 0.0,
                lambda: self.dPdx_history[step],
            )
        else:
            dPdx = 0.0
        nse.set_initial_field("velocity_hat", init_field)
        nse.dPdx = dPdx
        self.dPdx_fwd = dPdx
        nse.end_time = -1 * self.get_dt() * self.number_of_inner_steps
        velocity_u_hat_history, current_dPdx_history, _ = nse.solve_scan()
        current_velocity_field_u_history = cast("jnp_array", velocity_u_hat_history)
        jax.clear_caches()  # type: ignore
        gc.collect()
        return current_velocity_field_u_history, current_dPdx_history

    def run_backward_calculation(self) -> None:
        jax.clear_caches()  # type: ignore
        gc.collect()
        if not self.is_backward_calculation_done():
            self.run_forward_calculation()
            self.before_time_step_fn = None
            self.after_time_step_fn = None
            self.write_entire_output = False
            self.write_intermediate_output = False
            self.activate_jit()
            assert self.dPdx_history is not None
            if self.constant_mass_flux:
                self.dPdx = -self.dPdx_history[-2]
            else:
                self.dPdx = 0.0
            # self.dPdx = self.dPdx_history[-2] # TODO
            print_verb("performing backward (adjoint) calculation...")
            jax.clear_caches()  # type: ignore
            gc.collect()
            self.solve_scan()

    def get_gain(self) -> float:
        self.run_forward_calculation()
        u_0 = self.forward_equation.get_initial_field("velocity_hat").no_hat()
        u_T = self.forward_equation.get_latest_field("velocity_hat").no_hat()
        self.gain = u_T.energy() / u_0.energy()
        return self.gain

    def get_dissipation_average(self) -> float:
        self.run_forward_calculation()
        if self.checkpointing:
            print_verb(
                "warning: dissipation calculation may be inaccurate. This does not affect the optimisation, however."
            )
        u_hat_hist = self.velocity_field_u_history
        assert u_hat_hist is not None
        dissipation_hat = FourierField.Zeros(self.get_physical_domain())
        domain = self.get_domain()
        for u_hat in u_hat_hist:
            for i in range(3):
                for j in range(3):
                    dissipation_hat += domain.diff(u_hat[i], j) ** 2
        dissipation_hat /= np.abs(self.get_Re_tau() * len(u_hat_hist))
        return dissipation_hat.no_hat().volume_integral()

    def get_grad(self) -> "jnp_array":
        self.run_backward_calculation()
        u_hat_0 = self.forward_equation.get_initial_field("velocity_hat")
        v_hat_0 = self.get_latest_field("velocity_hat")
        gain = self.get_gain()
        e_0 = u_hat_0.no_hat().energy()

        return (gain * u_hat_0.get_data() - v_hat_0.get_data()) / e_0

    def get_projected_grad_from_u_and_v(
        self,
        step_size: float,
        u_hat_0: VectorField["FourierField"],
        v_hat_0: VectorField["FourierField"],
    ) -> Tuple["jnp_array", bool]:
        e_0 = u_hat_0.no_hat().energy()
        lam = -1.0

        def get_new_energy_0(l: float) -> float:
            return (
                ((1 + step_size * l) * u_hat_0 - step_size / self.gain * v_hat_0)
                .no_hat()
                .energy()
            )

        print_verb("optimising lambda...", verbosity_level=2)
        i = 0
        max_iter = 100
        tol = 1e-25  # can be fairly high as we normalize the result anyway
        while abs(get_new_energy_0(lam) - e_0) / e_0 > tol and i < max_iter:
            lam += -(get_new_energy_0(lam) - e_0) / jax.grad(get_new_energy_0)(lam)
            i += 1
        print_verb(
            "optimising lambda done in",
            i,
            "iterations, lambda:",
            lam,
            verbosity_level=2,
        )
        print_verb("energy:", get_new_energy_0(lam), verbosity_level=2)

        return (
            (lam * u_hat_0.get_data() - v_hat_0.get_data() / self.gain),
            i < max_iter,
        )

    def get_projected_grad(self, step_size: float) -> Tuple["jnp_array", bool]:
        self.run_backward_calculation()
        u_hat_0 = self.velocity_u_hat_0
        v_hat_0 = self.get_latest_field("velocity_hat")
        return self.get_projected_grad_from_u_and_v(step_size, u_hat_0, v_hat_0)

    def get_projected_cg_grad(
        self, step_size: float, beta: float, old_grad: "jnp_array"
    ) -> Tuple["jnp_array", bool]:
        self.run_backward_calculation()
        u_hat_0 = self.velocity_u_hat_0
        v_hat_0 = self.get_latest_field("velocity_hat")
        e_0 = u_hat_0.no_hat().energy()
        lam = -1.0

        def get_new_energy_0(l: float) -> float:
            return (
                (
                    (1 + step_size * l) * u_hat_0
                    + step_size / self.gain * (-1 * v_hat_0 + beta * old_grad)
                )
                .no_hat()
                .energy()
            )

        print_verb("optimising lambda...")
        i = 0
        max_iter = 100
        tol = 1e-25  # can be fairly high as we normalize the result anyway
        while abs(get_new_energy_0(lam) - e_0) / e_0 > tol and i < max_iter:
            lam += -(get_new_energy_0(lam) - e_0) / jax.grad(get_new_energy_0)(lam)
            i += 1
        print_verb(
            "optimising lambda done in",
            i,
            "iterations, lambda:",
            lam,
            verbosity_level=2,
        )
        print_verb("energy:", get_new_energy_0(lam), verbosity_level=2)

        return (
            (
                lam * u_hat_0.get_data()
                + (-1 * v_hat_0.get_data() + beta * old_grad) / self.gain
            ),
            i < max_iter,
        )


def perform_step_navier_stokes_perturbation_dual(
    nse: NavierStokesVelVortPerturbation,
    step_size: float = 1e-2,
) -> Tuple[jsd_float, "jnp_array"]:

    nse_dual = NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
        nse
    )
    return (nse_dual.get_gain(), nse_dual.get_projected_grad(step_size)[0])

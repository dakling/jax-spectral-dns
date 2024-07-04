#!/usr/bin/env python3

from __future__ import annotations

from jax_spectral_dns.navier_stokes import (
    get_div_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
    helicity_to_nonlinear_terms,
)

NoneType = type(None)
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

from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
from jax_spectral_dns.navier_stokes import (
    get_nabla_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
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
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vel_u_hat = (
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
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
    hel_new_hat = get_helicity_perturbation_dual_convection(
        physical_domain,
        fourier_domain,
        vel_v_hat_new,  # v
        vel_u_hat,  # U
        vel_v_new,  # v
        vel_u,  # U
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


def get_helicity_perturbation_dual_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    linearize: bool = False,
) -> "jnp_array":
    vel_u_hat = (
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
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
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    hel_new_hat = get_helicity_perturbation_dual_diffusion(
        physical_domain,
        fourier_domain,
        vel_v_hat_new,  # v
        vel_u_base_hat,  # U
        vel_small_u_hat,  # u
        linearize,
    )

    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


def update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    vel_v_new = jnp.array(
        [
            fourier_domain.field_no_hat(vel_v_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    vel_u_hat = (
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
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
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_v_hat_new)


def update_nonlinear_terms_high_performance_perturbation_dual_rotational(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_v_hat_new: "jnp_array",  # v
    vel_u_base_hat: "jnp_array",  # U
    vel_small_u_hat: "jnp_array",  # u
    linearize: bool = False,
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vel_u_hat = (
        jax.lax.cond(linearize, lambda: 0.0, lambda: 1.0) * vel_small_u_hat
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
        else:
            print_verb("not using checkpointing")
        self.forward_equation.activate_jit()

    def set_linearize(self, lin: bool) -> None:
        self.linearize = lin
        velocity_base_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_base_hat"
        )
        # self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual_rotational(
        self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            velocity_base_hat.get_data(),
            self.get_velocity_u_hat(t),
            linearize=self.linearize,
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

        nse_dual = cls(
            None,
            nse,
            Re_tau=-Re_tau,
            dt=-dt,
            end_time=-end_time,
            velocity_base_hat=nse.get_latest_field("velocity_base_hat"),
            constant_mass_flux=nse.constant_mass_flux**params,
        )
        nse_dual.set_linearize(nse.linearize)
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

    def update_with_nse(self) -> None:
        self.forward_equation.write_entire_output = True
        self.forward_equation.write_intermediate_output = False
        self.clear_field("velocity_hat")
        self.velocity_field_u_history = None

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

    def solve_scan(self) -> Tuple[Union["jnp_array", VectorField[FourierField]], int]:
        if not self.checkpointing:
            return super().solve_scan()
        else:
            nse = self.forward_equation
            self.number_of_time_steps = nse.number_of_time_steps
            self.number_of_outer_steps = nse.number_of_outer_steps
            self.number_of_inner_steps = nse.number_of_inner_steps

            def get_inner_step_fn(
                current_velocity_field_u_history: "jnp_array",
            ) -> Callable[
                [Tuple["jnp_array", int], Any], Tuple[Tuple["jnp_array", int], None]
            ]:
                def inner_step_fn(
                    u0: Tuple["jnp_array", int], _: Any
                ) -> Tuple[Tuple["jnp_array", int], None]:
                    u0_, time_step = u0
                    self.current_velocity_field_u_history = (
                        current_velocity_field_u_history
                    )
                    out = self.perform_time_step(u0_, time_step)
                    self.current_velocity_field_u_history = None

                    return ((out, time_step + 1), None)

                return inner_step_fn

            def step_fn(
                u0: Tuple["jnp_array", int], _: Any
            ) -> Tuple[Tuple["jnp_array", int], Tuple["jnp_array", int]]:
                timestep = u0[1]
                outer_start_step = timestep // self.number_of_inner_steps
                self.current_u_history_start_step = outer_start_step
                current_velocity_field_u_history = (
                    self.run_forward_calculation_subrange(outer_start_step)
                )
                out, _ = jax.lax.scan(
                    jax.checkpoint(get_inner_step_fn(current_velocity_field_u_history)),  # type: ignore[attr-defined]
                    u0,
                    xs=None,
                    length=self.number_of_inner_steps,
                    # inner_step_fn, u0, xs=None, length=number_of_inner_steps
                )
                return out, out

            u0 = self.get_initial_field("velocity_hat").get_data()
            ts = jnp.arange(0, self.end_time, self.get_dt())

            if self.write_intermediate_output and not self.write_entire_output:
                u_final, trajectory = jax.lax.scan(
                    step_fn, (u0, 0), xs=None, length=self.number_of_outer_steps
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
                return (trajectory[0], len(ts))
            elif self.write_entire_output:
                u_final, trajectory = jax.lax.scan(
                    step_fn, (u0, 0), xs=None, length=self.number_of_outer_steps
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
                return (trajectory[0], len(ts))
            else:
                u_final, _ = jax.lax.scan(
                    step_fn, (u0, 0), xs=None, length=self.number_of_outer_steps
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
                return (velocity_final, len(ts))

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
            velocity_u_hat_history_, _ = nse.solve_scan()
            iteration_duration = time.time() - start_time
            try:
                print_verb(
                    "forward calculation took", format_timespan(iteration_duration)
                )
            except Exception:
                print_verb("forward calculation took", iteration_duration, "seconds")
            self.velocity_field_u_history = cast("jnp_array", velocity_u_hat_history_)
        self.set_initial_field(
            "velocity_hat", -1 * nse.get_latest_field("velocity_hat")
        )
        self.forward_equation = nse  # not sure if this is necessary

    def run_forward_calculation_subrange(self, outer_timestep: int) -> "jnp_array":
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
        nse.set_initial_field("velocity_hat", init_field)
        nse.end_time = -1 * self.get_dt() * self.number_of_inner_steps
        velocity_u_hat_history_, _ = nse.solve_scan()
        current_velocity_field_u_history = cast("jnp_array", velocity_u_hat_history_)
        return current_velocity_field_u_history

    def run_backward_calculation(self) -> None:
        if not self.is_backward_calculation_done():
            self.run_forward_calculation()
            self.before_time_step_fn = None
            self.after_time_step_fn = None
            self.write_entire_output = False
            self.write_intermediate_output = False
            self.activate_jit()
            print_verb("performing backward (adjoint) calculation...")
            self.solve()

    def get_gain(self) -> float:
        self.run_forward_calculation()
        u_0 = self.forward_equation.get_initial_field("velocity_hat").no_hat()
        u_T = self.forward_equation.get_latest_field("velocity_hat").no_hat()
        self.gain = u_T.energy() / u_0.energy()
        return self.gain

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

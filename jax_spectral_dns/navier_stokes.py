#!/usr/bin/env python3
from __future__ import annotations

from numpy.polynomial.chebyshev import Chebyshev

from jax_spectral_dns.cheb import cheby

NoneType = type(None)
import os
from operator import rshift
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union, cast
from typing_extensions import Self
import jax
import jax.numpy as jnp
from matplotlib import axes
import numpy as np
from functools import partial, reduce
import matplotlib.figure as figure
from matplotlib.axes import Axes
import h5py  # type: ignore

# from importlib import reload
import sys

from jax_spectral_dns.domain import PhysicalDomain, FourierDomain
from jax_spectral_dns.field import (
    Field,
    PhysicalField,
    VectorField,
    FourierField,
    FourierFieldSlice,
)
from jax_spectral_dns.equation import Equation, E, print_verb
from jax_spectral_dns.fixed_parameters import NavierStokesVelVortFixedParameters
from jax_spectral_dns.linear_stability_calculation import LinearStabilityCalculation

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        np_float_array,
        np_complex_array,
        jsd_float,
        jnp_array,
        np_jnp_array,
        Vel_fn_type,
    )


def helicity_to_nonlinear_terms(
    fourier_domain: FourierDomain,
    hel_new_hat: "jnp_array",
    vel_hat_new: "jnp_array",
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
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
    conv_ns_hat_new = -hel_new_hat
    vort_hat_new = fourier_domain.curl(vel_hat_new)

    return (
        h_v_hat_new,
        h_g_hat_new,
        jnp.array(vort_hat_new),
        jnp.array(conv_ns_hat_new),
    )


def get_vel_1_nabla_vel_2(
    fourier_domain: FourierDomain,
    vel_1: "jnp_array",
    vel_2_hat: "jnp_array",
) -> "jnp_array":
    vel_1_nabla_vel_2 = jnp.zeros_like(vel_1)
    for i in fourier_domain.all_dimensions():
        for j in fourier_domain.all_dimensions():
            nabla_vel_2_hat = fourier_domain.diff(vel_2_hat[i], j)
            nabla_vel_2 = fourier_domain.field_no_hat(nabla_vel_2_hat)
            vel_u_i_nabla_u_j = vel_1[j] * nabla_vel_2
            vel_1_nabla_vel_2 = vel_1_nabla_vel_2.at[i].set(
                vel_1_nabla_vel_2.at[i].get() + vel_u_i_nabla_u_j
            )
    return vel_1_nabla_vel_2


def get_helicity_convection(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
    vel_new: "jnp_array",
) -> "jnp_array":

    vel_new_nabla_vel_new = get_vel_1_nabla_vel_2(fourier_domain, vel_new, vel_hat_new)
    # n = fourier_domain.number_of_dimensions
    # # TODO can I make this more efficient?
    # dvel_hat_i_dx_j = jnp.array(
    #     [[fourier_domain.diff(vel_hat_new[i], j) for j in range(n)] for i in range(n)]
    # )
    # vel_new_nabla_vel_new = jnp.sum(
    #     jnp.array(
    #         [
    #             [
    #                 vel_new[j] * fourier_domain.field_no_hat(dvel_hat_i_dx_j[i, j])
    #                 for j in range(n)
    #             ]
    #             for i in range(n)
    #         ]
    #     ),
    #     axis=1,
    # )
    hel_new = -vel_new_nabla_vel_new

    hel_new_hat = jnp.array(
        [
            physical_domain.field_hat(hel_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    return hel_new_hat


def get_helicity_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
) -> "jnp_array":
    vel_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new[i])
            # )
            fourier_domain.field_no_hat(vel_hat_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )

    nabla_vel_new_vel_new_hat = get_nabla_vel_1_vel_2(
        physical_domain, fourier_domain, vel_new, vel_new, vel_hat_new
    )
    # n = fourier_domain.number_of_dimensions
    # nabla_vel_new_vel_new_hat = jnp.sum(
    #     jnp.array(
    #         [
    #             [
    #                 fourier_domain.diff(
    #                     physical_domain.field_hat(vel_new[i] * vel_new[j]), j
    #                 )
    #                 for j in range(n)
    #             ]
    #             for i in range(n)
    #         ]
    #     ),
    #     axis=1,
    # )
    # nabla_vel_new_vel_new_hat = []
    # for i in physical_domain.all_dimensions():
    #     nabla_vel_new_vel_new_i_hat = jnp.zeros_like(vel_hat_new[0])
    #     for j in physical_domain.all_dimensions():
    #         # the nonlinear term
    #         vel_u_i_u_j = vel_new[i] * vel_new[j]
    #         vel_u_i_u_j_hat = physical_domain.field_hat(vel_u_i_u_j)
    #         nabla_vel_new_vel_new_i_hat += fourier_domain.diff(vel_u_i_u_j_hat, j)
    #     nabla_vel_new_vel_new_hat.append(nabla_vel_new_vel_new_i_hat)
    # nabla_vel_new_vel_new_hat_ = jnp.array(nabla_vel_new_vel_new_hat)
    hel_new_hat = -nabla_vel_new_vel_new_hat
    return hel_new_hat


def update_nonlinear_terms_high_performance_convection(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
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
    hel_new_hat = get_helicity_convection(
        physical_domain, fourier_domain, vel_hat_new, vel_new
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


def get_nabla_vel_1_vel_2(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_1: "jnp_array",
    vel_2: "jnp_array",
    vel_1_hat: "jnp_array",
) -> "jnp_array":
    nabla_vel_1_vel_2_hat = jnp.zeros_like(vel_1_hat)
    for i in physical_domain.all_dimensions():
        for j in physical_domain.all_dimensions():
            vel_u_i_u_j = vel_1[i] * vel_2[j]
            vel_u_i_u_j_hat = physical_domain.field_hat(vel_u_i_u_j)
            nabla_vel_1_vel_2_hat = nabla_vel_1_vel_2_hat.at[i].set(
                nabla_vel_1_vel_2_hat.at[i].get()
                + fourier_domain.diff(vel_u_i_u_j_hat, j)
            )
    return nabla_vel_1_vel_2_hat


def get_div_vel_1_vel_2(
    fourier_domain: FourierDomain,
    vel_1_hat: "jnp_array",
    vel_2: "jnp_array",
) -> "jnp_array":
    div_vel_1_hat = fourier_domain.divergence(vel_1_hat)
    div_vel_1 = fourier_domain.field_no_hat(div_vel_1_hat)
    out = jnp.array([div_vel_1 * vel_2[i] for i in fourier_domain.all_dimensions()])
    return out


def update_nonlinear_terms_high_performance_diffusion(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:
    hel_new_hat = get_helicity_diffusion(physical_domain, fourier_domain, vel_hat_new)
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


def update_nonlinear_terms_high_performance_skew_symmetric(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
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
    div_vel_new_vel_new = get_div_vel_1_vel_2(fourier_domain, vel_hat_new, vel_new)
    div_vel_new_vel_new_hat = jnp.array(
        [
            physical_domain.field_hat(div_vel_new_vel_new[i])
            for i in physical_domain.all_dimensions()
        ]
    )
    hel_new_hat = (
        get_helicity_convection(physical_domain, fourier_domain, vel_hat_new, vel_new)
        + 0.5 * div_vel_new_vel_new_hat
    )
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


def update_nonlinear_terms_high_performance_rotational(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vort_hat_new = fourier_domain.curl(vel_hat_new)
    vel_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vel_hat_new.at[i].get())
            # )
            fourier_domain.field_no_hat(vel_hat_new.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
        ]
    )
    vort_new = jnp.array(
        [
            # fourier_domain.filter_field_nonfourier_only(
            #     fourier_domain.field_no_hat(vort_hat_new[i])
            # )
            fourier_domain.field_no_hat(vort_hat_new[i])
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
    return helicity_to_nonlinear_terms(fourier_domain, hel_new_hat, vel_hat_new)


class NavierStokesVelVort(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation)"

    def __init__(self, velocity_field: VectorField[FourierField], **params: Any):
        if "physical_domain" in params:
            physical_domain = params["physical_domain"]
            domain = physical_domain.hat()
        else:
            domain = velocity_field[0].fourier_domain
            physical_domain = velocity_field[0].physical_domain

        max_cfl = params.get("max_cfl", 0.7)

        Re_tau = self.get_Re_tau(**params)
        self.nonlinear_update_fn: Callable[
            ["jnp_array", int],
            Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"],
            # ] = lambda vel, _: update_nonlinear_terms_high_performance_rotational(
        ] = lambda vel, _: update_nonlinear_terms_high_performance_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
        )
        super().__init__(domain, velocity_field, **params)

        n_rk_steps = 3

        (
            poisson_mat,
            rk_mats_rhs,
            rk_mats_lhs_inv,
            rk_rhs_inhom,
            rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns,
        ) = self.assemble_matrices(**params)

        self.nse_fixed_parameters = NavierStokesVelVortFixedParameters(
            physical_domain=physical_domain,
            poisson_mat=poisson_mat,
            rk_mats_rhs=rk_mats_rhs,
            rk_mats_lhs_inv=rk_mats_lhs_inv,
            rk_rhs_inhom=rk_rhs_inhom,
            rk_mats_lhs_inv_inhom=rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns=rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns=rk_mats_lhs_inv_ns,
            Re_tau=self.get_Re_tau(**params),
            max_cfl=max_cfl,
            u_max_over_u_tau=self.get_u_max_over_u_tau(**params),
            number_of_rk_steps=n_rk_steps,
        )
        print_verb("using RK" + str(n_rk_steps) + " time stepper", verbosity_level=2)

        self.constant_mass_flux = params.get("constant_mass_flux", False)
        if self.constant_mass_flux:
            if not params.get("non_verbose", False):
                print_verb("enforcing constant mass flux", verbosity_level=2)
            self.flow_rate = self.get_flow_rate()
            self.dPdx = -self.flow_rate * 3 / 2 / self.get_Re_tau()
        else:
            if not params.get("non_verbose", False):
                print_verb("enforcing constant pressure gradient", verbosity_level=2)
            self.flow_rate = self.get_flow_rate()
            self.dPdx = -1.0

        print_verb("calculated flow rate: ", self.flow_rate, verbosity_level=3)

        cont_error = jnp.sqrt(velocity_field.no_hat().div().energy())
        print_verb(
            "continuity error of initial condition:", cont_error, verbosity_level=2
        )
        self.source_x_00 = None
        self.source_z_00 = None

    @classmethod
    def FromDomain(cls, domain: PhysicalDomain, **params: Any) -> Self:
        velocity_field: VectorField[PhysicalField] = VectorField.Zeros(
            PhysicalField, domain
        )
        velocity_field_hat = velocity_field.hat()
        velocity_field_hat.name = "velocity_hat"
        return cls(velocity_field_hat, **params)

    @classmethod
    def FromVelocityField(
        cls, velocity_field: VectorField[PhysicalField], **params: Any
    ) -> Self:
        velocity_field_hat = velocity_field.hat()
        velocity_field_hat.name = "velocity_hat"
        return cls(velocity_field_hat, **params)

    @classmethod
    def FromRandom(cls, shape: Tuple[int, ...], **params: Any) -> Self:
        domain = PhysicalDomain.create(shape, (True, False, True))
        vel_x = PhysicalField.FromRandom(domain, name="u0")
        vel_y = PhysicalField.FromRandom(domain, name="u1")
        vel_z = PhysicalField.FromRandom(domain, name="u2")
        vel = VectorField([vel_x, vel_y, vel_z], "velocity")
        return cls.FromVelocityField(vel, **params)

    def get_domain(self) -> FourierDomain:
        out: FourierDomain = super().get_domain()  # type: ignore[assignment]
        return out

    def get_physical_domain(self) -> PhysicalDomain:
        return self.nse_fixed_parameters.physical_domain

    def get_field(self, name: str, index: int) -> "VectorField[FourierField]":
        out = cast(VectorField[FourierField], super().get_field(name, index))
        return out

    def get_fields(self, name: str) -> List["VectorField[FourierField]"]:
        return cast(List[VectorField[FourierField]], super().get_fields(name))

    def get_initial_field(self, name: str) -> "VectorField[FourierField]":
        out = cast(VectorField[FourierField], super().get_initial_field(name))
        return out

    def get_latest_field(self, name: str) -> "VectorField[FourierField]":
        out = cast(VectorField[FourierField], super().get_latest_field(name))
        return out

    def get_fixed_params(self) -> "NavierStokesVelVortFixedParameters":
        return self.nse_fixed_parameters

    def assemble_matrices(
        self, **params: Any
    ) -> Tuple[Optional["np_complex_array"], ...]:
        n_rk_steps = 3
        domain = self.get_domain()
        calculation_size = self.end_time / self.get_dt()
        for i in self.all_dimensions():
            calculation_size *= self.get_domain().number_of_cells(i)
        prepare_matrices_threshold = params.get("prepare_matrices_threshold", 1e12)
        self.prepare_matrices = params.get(
            "prepare_matrices", calculation_size < prepare_matrices_threshold
        )
        if self.prepare_matrices:
            print_verb("preparing differentiation matrices...", verbosity_level=2)
            poisson_mat = domain.assemble_poisson_matrix()
            (
                rk_mats_rhs,
                rk_mats_lhs_inv,
                rk_rhs_inhom,
                rk_mats_lhs_inv_inhom,
                rk_mats_rhs_ns,
                rk_mats_lhs_inv_ns,
            ) = self.prepare_assemble_rk_matrices(
                domain, self.get_Re_tau(**params), self.get_dt(), n_rk_steps
            )

            poisson_mat.setflags(write=False)
            rk_mats_rhs.setflags(write=False)
            rk_mats_lhs_inv.setflags(write=False)
            rk_rhs_inhom.setflags(write=False)
            rk_mats_lhs_inv_inhom.setflags(write=False)
            rk_mats_rhs_ns.setflags(write=False)
            rk_mats_lhs_inv_ns.setflags(write=False)
            print_verb("done preparing differentiation matrices", verbosity_level=2)
        else:
            print_verb(
                "not preparing differentiation matrices - \
                this may reduce memory usage but can carry \
                a significant runtime penalty!",
                verbosity_level=1,
            )
            poisson_mat = None
            (
                rk_mats_rhs,
                rk_mats_lhs_inv,
                rk_rhs_inhom,
                rk_mats_lhs_inv_inhom,
                rk_mats_rhs_ns,
                rk_mats_lhs_inv_ns,
            ) = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return (
            poisson_mat,
            rk_mats_rhs,
            rk_mats_lhs_inv,
            rk_rhs_inhom,
            rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns,
        )

    # @partial(jax.jit, static_argnums=0)
    def get_poisson_mat(self) -> "np_complex_array":
        if self.prepare_matrices:
            return cast("np_complex_array", self.get_fixed_params().poisson_mat)
        else:
            return self.get_domain().assemble_poisson_matrix()

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_rk_mats_lhs_inv(self, step: int, kx: int, kz: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_mats_lhs_inv)[step, kx, kz],
            )
        else:
            dt = self.get_dt()
            kx_ = jnp.asarray(self.get_domain().grid[0])[kx]
            kz_ = jnp.asarray(self.get_domain().grid[2])[kz]
            domain = self.get_domain()
            physical_domain = self.get_physical_domain()
            Re_tau = self.get_Re_tau()
            _, beta, _, _ = self.get_rk_parameters()
            D2 = np.linalg.matrix_power(physical_domain.diff_mats[1], 2)
            Ly = 1 / Re_tau * D2
            n = Ly.shape[0]
            I = np.eye(n)
            L = Ly + I * (-(kx_**2 + kz_**2)) / Re_tau
            rk_mat_lhs = I - beta[step] * dt * L
            rk_mat_lhs_ = domain.enforce_homogeneous_dirichlet_jnp(rk_mat_lhs)
            return cast("np_jnp_array", jnp.linalg.inv(rk_mat_lhs_))

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_rk_mats_rhs(self, step: int, kx: int, kz: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_mats_rhs)[step, kx, kz],
            )
        else:
            dt = self.get_dt()
            kx_ = jnp.asarray(self.get_domain().grid[0])[kx]
            kz_ = jnp.asarray(self.get_domain().grid[2])[kz]
            physical_domain = self.get_physical_domain()
            Re_tau = self.get_Re_tau()
            alpha, _, _, _ = self.get_rk_parameters()
            D2 = np.linalg.matrix_power(physical_domain.diff_mats[1], 2)
            Ly = 1 / Re_tau * D2
            n = Ly.shape[0]
            I = np.eye(n)
            L = Ly + I * (-(kx_**2 + kz_**2)) / Re_tau
            rk_mat_rhs = I + alpha[step] * dt * L
            return rk_mat_rhs

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_rk_mats_lhs_inv_inhom(self, step: int, kx: int, kz: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_mats_lhs_inv_inhom)[
                    step, kx, kz
                ],
            )
        else:
            dt = self.get_dt()
            kx_ = jnp.asarray(self.get_domain().grid[0])[kx]
            kz_ = jnp.asarray(self.get_domain().grid[2])[kz]
            physical_domain = self.get_physical_domain()
            domain = self.get_domain()
            Re_tau = self.get_Re_tau()
            _, beta, _, _ = self.get_rk_parameters()
            D2 = np.linalg.matrix_power(physical_domain.diff_mats[1], 2)
            Ly = 1 / Re_tau * D2
            n = Ly.shape[0]
            I = np.eye(n)
            L = Ly + I * (-(kx_**2 + kz_**2)) / Re_tau
            rhs_inhom = np.zeros(n)
            lhs_mat_inhom = I - beta[step] * dt * L
            (
                lhs_mat_inhom_,
                _,
            ) = domain.enforce_inhomogeneous_dirichlet_jnp(
                lhs_mat_inhom, rhs_inhom, 0.0, 1.0
            )
            lhs_mat_inv_inhom = jnp.linalg.inv(lhs_mat_inhom_)
            return cast("np_jnp_array", lhs_mat_inv_inhom)

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_rk_rhs_inhom(self, step: int, kx: int, kz: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_rhs_inhom)[step, kx, kz],
            )
        else:
            dt = self.get_dt()
            kx_ = jnp.asarray(self.get_domain().grid[0])[kx]
            kz_ = jnp.asarray(self.get_domain().grid[2])[kz]
            physical_domain = self.get_physical_domain()
            domain = self.get_domain()
            Re_tau = self.get_Re_tau()
            _, beta, _, _ = self.get_rk_parameters()
            D2 = np.linalg.matrix_power(physical_domain.diff_mats[1], 2)
            Ly = 1 / Re_tau * D2
            n = Ly.shape[0]
            I = np.eye(n)
            L = Ly + I * (-(kx_**2 + kz_**2)) / Re_tau
            rhs_inhom = np.zeros(n)
            lhs_mat_inhom = I - beta[step] * dt * L
            (
                _,
                rhs_inhom_,
            ) = domain.enforce_inhomogeneous_dirichlet_jnp(
                lhs_mat_inhom, rhs_inhom, 0.0, 1.0
            )
            return rhs_inhom_

    # @partial(jax.jit, static_argnums=(0, 1))
    def get_rk_mats_lhs_inv_ns(self, step: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_mats_lhs_inv_ns)[step],
            )
        else:
            dt = self.get_dt()
            _, beta, _, _ = self.get_rk_parameters()
            Re_tau = self.get_Re_tau()
            D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
            n = D2_hom_diri.shape[0]
            Z = np.zeros((n, n))
            L_NS_y = 1 / Re_tau * np.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])
            I_ns = np.eye(2 * n)
            L_ns = L_NS_y + I_ns * (-(0**2 + 0**2)) / Re_tau
            lhs_mat_ns = I_ns - beta[step] * dt * L_ns
            lhs_mat_inv_ns = jnp.linalg.inv(lhs_mat_ns)
            return cast("np_jnp_array", lhs_mat_inv_ns)

    # @partial(jax.jit, static_argnums=(0, 1))
    def get_rk_mats_rhs_ns(self, step: int) -> "np_jnp_array":
        if self.prepare_matrices:
            return cast(
                "np_jnp_array",
                jnp.asarray(self.get_fixed_params().rk_mats_rhs_ns)[step],
            )
        else:
            dt = self.get_dt()
            alpha, _, _, _ = self.get_rk_parameters()
            Re_tau = self.get_Re_tau()
            D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
            n = D2_hom_diri.shape[0]
            Z = np.zeros((n, n))
            L_NS_y = 1 / Re_tau * np.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])
            I_ns = np.eye(2 * n)
            L_ns = L_NS_y + I_ns * (-(0**2 + 0**2)) / Re_tau
            rhs_mat_ns = I_ns + alpha[step] * dt * L_ns
            return rhs_mat_ns

    def get_Re_tau(self, **params: Any) -> "float":
        try:
            return self.nse_fixed_parameters.Re_tau
        except Exception:
            try:
                Re_tau = params["Re_tau"]
            except KeyError:
                try:
                    Re_tau = params["Re"] / self.get_u_max_over_u_tau(**params)
                except KeyError:
                    raise Exception(
                        "Either Re or Re_tau has to be given as a parameter."
                    )
            return cast("float", Re_tau)

    def get_max_cfl(self) -> "float":
        return self.nse_fixed_parameters.max_cfl

    def get_dt_update_frequency(self) -> "int":
        return self.nse_fixed_parameters.dt_update_frequency

    def get_u_max_over_u_tau(self, **params: Any) -> "float":
        try:
            return self.nse_fixed_parameters.u_max_over_u_tau
        except Exception:
            return cast("float", params.get("u_max_over_u_tau", 1.0))

    def get_number_of_rk_steps(self) -> "int":
        return self.nse_fixed_parameters.number_of_rk_steps

    def init_velocity(self, velocity_hat: VectorField[FourierField]) -> None:
        self.set_field("velocity_hat", 0, velocity_hat)

    def get_vorticity_and_helicity(
        self,
    ) -> Tuple[VectorField[FourierField], VectorField[FourierField]]:
        velocity_field_hat: VectorField[FourierField] = self.get_latest_field(
            "velocity_hat"
        )
        vort_hat = velocity_field_hat.curl()
        for i in jnp.arange(3):
            vort_hat[i].name = "vort_hat_" + str(i)

        hel_hat = velocity_field_hat.no_hat().cross_product(vort_hat.no_hat()).hat()
        for i in jnp.arange(3):
            hel_hat[i].name = "hel_hat_" + str(i)
        return (vort_hat, hel_hat)

    def get_flow_rate(
        self, vel_new_field_hat: Optional["jnp_array"] = None
    ) -> "jsd_float":

        if type(vel_new_field_hat) == NoneType:
            vel_hat: VectorField[FourierField] = self.get_latest_field("velocity_hat")
        else:
            assert vel_new_field_hat is not None
            vel_hat = VectorField.FromData(
                FourierField, self.get_physical_domain(), vel_new_field_hat
            )
        vel_hat_0: FourierField = vel_hat[0]
        int: PhysicalField = vel_hat_0.no_hat().definite_integral(1)  # type: ignore[assignment]
        return cast("jsd_float", int[0, 0])

    def update_velocity_field_data(self, vel_new_hat_field: "jnp_array") -> "jnp_array":
        if self.constant_mass_flux:
            current_flow_rate = self.get_flow_rate(vel_new_hat_field)
            flow_rate_diff = current_flow_rate - self.flow_rate
            return jnp.array(
                [
                    (
                        self.get_physical_domain().field_hat(
                            self.get_domain().field_no_hat(vel_new_hat_field[i])
                            + jnp.ones(self.get_physical_domain().get_shape_aliasing())
                            * (-1 * flow_rate_diff * 0.5)
                        )
                        if i == 0
                        else vel_new_hat_field[i]
                    )
                    for i in self.all_dimensions()
                ]
            )
        else:
            return vel_new_hat_field

    def update_velocity_field(
        self, vel_new_hat: "VectorField[FourierField]"
    ) -> "VectorField[FourierField]":
        if self.constant_mass_flux:
            return VectorField.FromData(
                FourierField,
                self.get_physical_domain(),
                self.update_velocity_field_data(vel_new_hat.get_data()),
                name="velocity_hat",
            )
        else:
            return vel_new_hat

    def update_pressure_gradient(
        self,
        vel_new_field_hat: Optional["jnp_array"] = None,
        dPdx: Optional["float"] = None,
    ) -> "float":
        if dPdx is None:
            dPdx_ = self.dPdx
        else:
            dPdx_ = dPdx
        if self.constant_mass_flux:
            current_flow_rate = self.get_flow_rate(vel_new_field_hat)

            flow_rate_diff = current_flow_rate - self.flow_rate
            dpdx_change = flow_rate_diff / self.get_dt()
            dPdx_ = dPdx_ + dpdx_change
            print_verb(
                "current flow rate:",
                current_flow_rate,
                verbosity_level=3,
                debug=Field.activate_jit_,
            )
            print_verb(
                "current flow rate deficit:",
                self.flow_rate - current_flow_rate,
                verbosity_level=3,
                debug=Field.activate_jit_,
            )
            print_verb(
                "current pressure gradient:",
                dPdx_,
                verbosity_level=3,
                debug=Field.activate_jit_,
            )
        else:
            self.flow_rate = self.get_flow_rate()
            print_verb(
                "current flow rate:",
                self.flow_rate,
                verbosity_level=3,
                debug=Field.activate_jit_,
            )
            print_verb(
                "current pressure gradient:",
                dPdx_,
                verbosity_level=3,
                debug=Field.activate_jit_,
            )
        return cast(float, dPdx_)

    def get_cheb_mat_2_homogeneous_dirichlet(self) -> "np_float_array":
        return self.get_domain().get_cheb_mat_2_homogeneous_dirichlet(1)

    def update_dt(self, new_dt: float) -> None:
        if self.prepare_matrices:
            print_verb("preparing differentiation matrices...", verbosity_level=2)
            poisson_mat = self.get_domain().assemble_poisson_matrix()
            (
                rk_mats_rhs,
                rk_mats_lhs_inv,
                rk_rhs_inhom,
                rk_mats_lhs_inv_inhom,
                rk_mats_rhs_ns,
                rk_mats_lhs_inv_ns,
            ) = self.prepare_assemble_rk_matrices(
                self.get_domain(),
                self.get_Re_tau(),
                new_dt,
                3,
            )

            poisson_mat.setflags(write=False)
            rk_mats_rhs.setflags(write=False)
            rk_mats_lhs_inv.setflags(write=False)
            rk_rhs_inhom.setflags(write=False)
            rk_mats_lhs_inv_inhom.setflags(write=False)
            rk_mats_rhs_ns.setflags(write=False)
            rk_mats_lhs_inv_ns.setflags(write=False)
            print_verb("done preparing differentiation matrices", verbosity_level=2)
        else:
            print_verb(
                "not preparing differentiation matrices - \
                this may reduce memory usage but can carry \
                a significant runtime penalty!",
                verbosity_level=1,
            )
            poisson_mat = None
            (
                rk_mats_rhs,
                rk_mats_lhs_inv,
                rk_rhs_inhom,
                rk_mats_lhs_inv_inhom,
                rk_mats_rhs_ns,
                rk_mats_lhs_inv_ns,
            ) = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        self.nse_fixed_parameters = NavierStokesVelVortFixedParameters(
            physical_domain=self.get_physical_domain(),
            poisson_mat=poisson_mat,
            rk_mats_rhs=rk_mats_rhs,
            rk_mats_lhs_inv=rk_mats_lhs_inv,
            rk_rhs_inhom=rk_rhs_inhom,
            rk_mats_lhs_inv_inhom=rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns=rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns=rk_mats_lhs_inv_ns,
            Re_tau=self.get_Re_tau(),
            max_cfl=self.get_max_cfl(),
            u_max_over_u_tau=self.get_u_max_over_u_tau(),
            number_of_rk_steps=3,
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
        vel = self.get_field("velocity_hat", i).no_hat()
        U = vel[0][1:, 1:, 1:]
        V = vel[1][1:, 1:, 1:]
        W = vel[2][1:, 1:, 1:]
        u_cfl = cast(float, (abs(DX) / abs(U)).min().real)
        v_cfl = cast(float, (abs(DY) / abs(V)).min().real)
        w_cfl = cast(float, (abs(DZ) / abs(W)).min().real)
        return self.get_dt() / jnp.array([u_cfl, v_cfl, w_cfl])

    def get_rk_parameters(
        self, n_steps: Optional[int] = None
    ) -> Tuple[List["jsd_float"], ...]:
        if n_steps is None:
            n_steps = self.get_number_of_rk_steps()
        if n_steps == 1:
            return (
                [1.0],
                [1.0],
                [1.0],
                [0.0],
            )
        elif n_steps == 3:
            return (
                [29 / 96, -3 / 40, 1 / 6],
                [37 / 160, 5 / 24, 1 / 6],
                [8 / 15, 5 / 12, 3 / 4],
                [0, -17 / 60, -5 / 12],
            )
        else:
            raise Exception("number of rk steps not supported")

    def prepare_assemble_rk_matrices(
        self,
        domain: FourierDomain,
        Re_tau: "jsd_float",
        dt: "jsd_float",
        number_of_rk_steps: int,
    ) -> Tuple["np_complex_array", ...]:
        alpha, beta, _, _ = self.get_rk_parameters(number_of_rk_steps)
        D2 = np.linalg.matrix_power(domain.diff_mats[1], 2)
        Ly = 1 / Re_tau * D2
        n = Ly.shape[0]
        I = np.eye(n)
        Z = np.zeros((n, n))
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        L_NS_y = 1 / Re_tau * np.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])

        rk_mats_rhs = np.zeros(
            (
                number_of_rk_steps,
                domain.number_of_cells(0),
                domain.number_of_cells(2),
                n,
                n,
            ),
            dtype=np.complex128,
        )
        rk_mats_lhs_inv = np.zeros(
            (
                number_of_rk_steps,
                domain.number_of_cells(0),
                domain.number_of_cells(2),
                n,
                n,
            ),
            dtype=np.complex128,
        )
        rk_rhs_inhom = np.zeros(
            (
                number_of_rk_steps,
                domain.number_of_cells(0),
                domain.number_of_cells(2),
                n,
            ),
            dtype=np.complex128,
        )
        rk_mats_lhs_inv_inhom = np.zeros(
            (
                number_of_rk_steps,
                domain.number_of_cells(0),
                domain.number_of_cells(2),
                n,
                n,
            ),
            dtype=np.complex128,
        )
        rk_mats_rhs_ns = np.zeros(
            (number_of_rk_steps, 2 * n, 2 * n), dtype=np.complex128
        )
        rk_mats_lhs_inv_ns = np.zeros(
            (number_of_rk_steps, 2 * n, 2 * n), dtype=np.complex128
        )
        for i in range(number_of_rk_steps):
            for xi, kx in enumerate(domain.grid[0]):
                for zi, kz in enumerate(domain.grid[2]):
                    L = Ly + I * (-(kx**2 + kz**2)) / Re_tau
                    rhs_mat = I + alpha[i] * dt * L
                    rhs_mat = domain.enforce_homogeneous_dirichlet(rhs_mat)
                    lhs_mat = I - beta[i] * dt * L
                    lhs_mat = domain.enforce_homogeneous_dirichlet(lhs_mat)
                    lhs_mat_inv = np.linalg.inv(lhs_mat)
                    rk_mats_rhs[i, xi, zi] = rhs_mat
                    rk_mats_lhs_inv[i, xi, zi] = lhs_mat_inv

                    rhs_inhom = np.zeros(n)

                    lhs_mat_inhom = I - beta[i] * dt * L
                    (
                        lhs_mat_inhom,
                        rhs_inhom,
                    ) = domain.enforce_inhomogeneous_dirichlet(
                        lhs_mat_inhom, rhs_inhom, 0.0, 1.0
                    )
                    lhs_mat_inv_inhom = np.linalg.inv(lhs_mat_inhom)
                    rk_rhs_inhom[i, xi, zi] = rhs_inhom
                    rk_mats_lhs_inv_inhom[i, xi, zi] = lhs_mat_inv_inhom

            I_ns = np.eye(2 * n)
            L_ns = L_NS_y + I_ns * (-(0**2 + 0**2)) / Re_tau
            rhs_mat_ns = I_ns + alpha[i] * dt * L_ns
            rhs_mat_ns = domain.enforce_homogeneous_dirichlet(rhs_mat_ns)
            lhs_mat_ns = I_ns - beta[i] * dt * L_ns
            lhs_mat_ns = domain.enforce_homogeneous_dirichlet(lhs_mat_ns)
            lhs_mat_inv_ns = np.linalg.inv(lhs_mat_ns)
            rk_mats_rhs_ns[i] = rhs_mat_ns
            rk_mats_lhs_inv_ns[i] = lhs_mat_inv_ns
        return (
            rk_mats_rhs,
            rk_mats_lhs_inv,
            rk_rhs_inhom,
            rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns,
        )

    def prepare(self) -> None:
        pass

    @classmethod
    def smooth_and_enforce_bc_vel_y(
        cls,
        physical_domain: PhysicalDomain,
        vel_y_slice: "jnp_array",
        M: Optional[int] = None,
    ) -> "jnp_array":
        def phi_n(n: int) -> Chebyshev:
            a = +(2 * (n + 2)) / (n + 1)
            b = -(n + 3) / (n + 1)
            arr = (
                np.eye(4 + n + 1)[4 + n].flatten()
                - a * np.eye(4 + n + 1)[2 + n].flatten()
                - b * np.eye(4 + n + 1)[n].flatten()
            )
            assert (1 - a - b) < 1e-10, "first cond too big, n: " + str(n)
            assert (
                (n + 4) ** 2 - a * (n + 2) ** 2 - b * n**2
            ) < 1e-10, "second cond too big, n: " + str(n)
            out: Chebyshev = Chebyshev(arr)
            return out

        N = vel_y_slice.shape[0]
        if M is None:
            M_ = N - 4
        else:
            M_ = M
        A = np.zeros((N, M_))
        ys = physical_domain.grid[1]
        for i in range(M_):
            for j in range(N):
                A[j, i] = phi_n(i)(ys[j])

        coeffs, _, _, _ = jnp.linalg.lstsq(A, vel_y_slice)
        return cast("jnp_array", A @ coeffs)

    @classmethod
    def vort_yvel_to_vel(
        cls,
        physical_domain: PhysicalDomain,
        vort: Optional["jnp_array"],
        vel_y: "jnp_array",
        vel_x_00: "jnp_array",
        vel_z_00: Optional["jnp_array"],
        two_d: bool = False,
    ) -> "jnp_array":
        domain = physical_domain.hat()
        # compute velocities in x and z directions
        number_of_input_arguments = 2
        Nx = domain.number_of_cells(0)
        Ny = domain.number_of_cells(1)
        Nz = domain.number_of_cells(2)

        if two_d:
            vort = jnp.zeros_like(vel_y)
            vel_z_00 = jnp.zeros_like(vel_x_00)
        assert vort is not None
        assert vel_z_00 is not None

        def rk_00() -> Tuple["jnp_array", ...]:
            return (
                (vel_x_00 * (1 + 0j)).astype(jnp.complex128),
                (
                    NavierStokesVelVort.smooth_and_enforce_bc_vel_y(
                        physical_domain,
                        vel_y[0, :, 0],  # 3 * Ny // 2
                    )
                    * (1 + 0j)
                ).astype(jnp.complex128),
                # (jnp.zeros_like(vel_x_00) * (1 + 0j)).astype(jnp.complex128),
                (vel_z_00 * (1 + 0j)).astype(jnp.complex128),
            )

        def rk_not_00(
            kx: int, kz: int, vort_: "jnp_array", vel_y_: "jnp_array"
        ) -> Tuple["jnp_array", ...]:
            kx_ = jnp.asarray(domain.grid[0])[kx]
            kz_ = jnp.asarray(domain.grid[2])[kz]
            j_kx = 1j * kx_
            j_kz = 1j * kz_
            minus_kx_kz_sq = -(kx_**2 + kz_**2)
            vel_y__ = NavierStokesVelVort.smooth_and_enforce_bc_vel_y(
                physical_domain,
                vel_y_,  # 3 * Ny // 2
            )
            vel_1_y_ = domain.diff_fourier_field_slice(vel_y__, 1, 1)
            vel_x_ = (-j_kx * vel_1_y_ + j_kz * vort_) / minus_kx_kz_sq
            if two_d:
                vel_z_ = jnp.zeros_like(vel_x_, dtype=jnp.complex128)
            else:
                vel_z_ = (-j_kz * vel_1_y_ - j_kx * vort_) / minus_kx_kz_sq

            return (
                vel_x_.astype(jnp.complex128),
                vel_y__.astype(jnp.complex128),
                vel_z_.astype(jnp.complex128),
            )

        def inner_map(kx: "jsd_float") -> Callable[["jnp_array"], "jnp_array"]:
            def fn(kz_one_pt_state: "jnp_array") -> "jnp_array":
                if two_d:
                    kz: int = 0
                else:
                    kz = cast(int, kz_one_pt_state[0].real.astype(int))
                fields_1d = jnp.split(
                    kz_one_pt_state[1:],
                    number_of_input_arguments,
                    axis=0,
                )
                # since the logical "and" causes problems for jax, we use arithmetic to decide if kx == kz == 0
                kx_and_kz_both_zero = (
                    jnp.exp(kx**2) * jnp.exp(kz**2) == 1
                )  # since kx and kz are integers, this can only be true if kx==kz==0

                out = jax.lax.cond(
                    kx_and_kz_both_zero,
                    lambda _, __: rk_00(),
                    lambda kx___, kz___: rk_not_00(kx___, kz___, *fields_1d),
                    cast(jnp.float64, kx.real).astype(int),
                    kz,
                )
                return cast("jnp_array", out)

            return fn

        def outer_map(kzs_: "np_jnp_array") -> Callable[["np_jnp_array"], "jnp_array"]:
            def fn(kx_state: "np_jnp_array") -> "jnp_array":
                kx = kx_state[0]
                # kx = kx_[0]
                fields_2d = jnp.split(
                    kx_state[1:],
                    number_of_input_arguments,
                    axis=0,
                    # state, number_of_input_arguments, axis=0
                )
                for i in range(len(fields_2d)):
                    fields_2d[i] = jnp.reshape(fields_2d[i], (Nz, Ny)).T
                state_slice = jnp.concatenate(fields_2d).T
                kz_state_slice = jnp.concatenate([kzs_.T, state_slice], axis=1)
                N = kz_state_slice.shape[0]
                batch_size = N // 1
                out: "jnp_array" = jax.lax.map(
                    inner_map(kx), kz_state_slice, batch_size=batch_size
                )
                # out: "jnp_array" = jax.vmap(inner_map(kx))(kz_state_slice)
                return out

            return fn

        # vel_y = domain.update_boundary_conditions(vel_y)

        # vort = domain.update_boundary_conditions(vort)

        kx_arr = np.atleast_2d(np.arange(Nx))
        kz_arr = np.atleast_2d(np.arange(Nz))
        state = jnp.concatenate(
            [
                jnp.moveaxis(vort, 1, 2),
                jnp.moveaxis(vel_y, 1, 2),
            ],
            axis=1,
        )
        kx_state = jnp.concatenate(
            [
                kx_arr.T,
                jnp.reshape(state, (Nx, (number_of_input_arguments * Ny * Nz))),
            ],
            axis=1,
        )
        N = kx_state.shape[0]
        batch_size = N // 1
        out: "jnp_array" = jax.lax.map(
            outer_map(kz_arr), kx_state, batch_size=batch_size
        )
        # out = jax.vmap(outer_map(kz_arr))(kx_state)
        u_v_w = [jnp.moveaxis(v, 1, 2) for v in out]
        return jnp.array([u_v_w[0], u_v_w[1], u_v_w[2]])

    def enforce_constant_mass_flux(
        self, vel_new_hat_field: "jnp_array", dPdx: "jsd_float", time_step: int
    ) -> Tuple["jnp_array", "jsd_float"]:
        if self.constant_mass_flux:
            dPdx = self.update_pressure_gradient(vel_new_hat_field, cast(float, dPdx))
            vel_new_hat_field = self.update_velocity_field_data(vel_new_hat_field)
        else:
            if Equation.verbosity_level >= 3:
                self.update_pressure_gradient(vel_new_hat_field, cast(float, dPdx))
        return vel_new_hat_field, dPdx

    def get_source_term(self, _: int) -> Optional["jnp_array"]:
        return None

    def perform_runge_kutta_step(
        self, vel_hat_data: "jnp_array", dPdx: "jsd_float", time_step: int
    ) -> Tuple["jnp_array", "jsd_float"]:

        # start runge-kutta stepping
        _, _, gamma, xi = self.get_rk_parameters()

        n = self.get_domain().number_of_cells(1)

        def perform_single_rk_step_for_single_wavenumber(
            step: int,
        ) -> Callable[
            [
                Tuple[int, int],
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
            ],
            Tuple["jnp_array", ...],
        ]:
            def fn(
                K: Tuple[int, int],
                v_1_lap_hat_sw: "jnp_array",
                vort_1_hat_sw: "jnp_array",
                h_v_hat_sw: "jnp_array",
                h_g_hat_sw: "jnp_array",
                h_v_hat_old_sw: "jnp_array",
                h_g_hat_old_sw: "jnp_array",
                v_0_hat_sw_00: "jnp_array",
                v_2_hat_sw_00: "jnp_array",
                conv_ns_hat_sw_0_00: "jnp_array",
                conv_ns_hat_sw_2_00: "jnp_array",
                conv_ns_hat_old_sw_0_00: "jnp_array",
                conv_ns_hat_old_sw_2_00: "jnp_array",
            ) -> Tuple["jnp_array", ...]:
                domain = self.get_domain()
                kx = K[0]
                kz = K[1]

                # wall-normal velocity
                # p-part
                lhs_mat_p_inv = jnp.asarray(self.get_rk_mats_lhs_inv(step, kx, kz))
                rhs_mat_p = jnp.asarray(self.get_rk_mats_rhs(step, kx, kz))

                TEST_MATRICES = "JAX_SPECTRAL_DNS_TEST_DIFF_MATS" in os.environ
                if TEST_MATRICES:
                    print_verb(
                        "WARNING: testing matrix assembly. Should only be done for testing purposes."
                    )
                    assert (
                        self.prepare_matrices == True
                    ), "this test only makes sense with prepare_matrices set to True"
                    self.prepare_matrices = False
                    lhs_mat_p_inv_ = jnp.asarray(self.get_rk_mats_lhs_inv(step, kx, kz))
                    rhs_mat_p_ = jnp.asarray(self.get_rk_mats_rhs(step, kx, kz))
                    jax.debug.print(
                        "lhs: {x}", x=jnp.linalg.norm(lhs_mat_p_inv - lhs_mat_p_inv_)
                    )
                    jax.debug.print(
                        "rhs: {x}", x=jnp.linalg.norm(rhs_mat_p - rhs_mat_p_)
                    )
                    self.prepare_matrices = True

                phi_hat_lap = v_1_lap_hat_sw

                N_p_new = h_v_hat_sw
                N_p_old = h_v_hat_old_sw

                rhs_p = (
                    rhs_mat_p @ phi_hat_lap
                    + (self.get_dt() * gamma[step]) * N_p_new
                    + (self.get_dt() * xi[step]) * N_p_old
                )
                # lhs_mat_p = domain.enforce_homogeneous_dirichlet(lhs_mat_p)
                rhs_p = domain.update_boundary_conditions_fourier_field_slice(rhs_p, 1)

                phi_hat_lap_new = lhs_mat_p_inv @ rhs_p

                v_1_lap_hat_new_p = phi_hat_lap_new

                # compute velocity in y direction
                v_1_lap_hat_new_p = (
                    domain.update_boundary_conditions_fourier_field_slice(
                        v_1_lap_hat_new_p, 1
                    )
                )
                v_1_hat_new_p = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_p, jnp.asarray(self.get_poisson_mat()), kx, kz
                )
                v_1_hat_new_p = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_p, 1
                )

                # a-part - numerical solution
                lhs_mat_a_inv = jnp.asarray(
                    self.get_rk_mats_lhs_inv_inhom(step, kx, kz)
                )
                rhs_a = jnp.asarray(self.get_rk_rhs_inhom(step, kx, kz))

                if TEST_MATRICES:
                    assert (
                        self.prepare_matrices == True
                    ), "this test only makes sense with prepare_matrices set to True"
                    self.prepare_matrices = False
                    lhs_mat_a_inv_ = jnp.asarray(
                        self.get_rk_mats_lhs_inv_inhom(step, kx, kz)
                    )
                    rhs_a_ = jnp.asarray(self.get_rk_rhs_inhom(step, kx, kz))
                    jax.debug.print(
                        "inhom lhs: {x}",
                        x=jnp.linalg.norm(lhs_mat_a_inv - lhs_mat_a_inv_),
                    )
                    jax.debug.print("inhom rhs: {x}", x=jnp.linalg.norm(rhs_a - rhs_a_))
                    self.prepare_matrices = True

                if TEST_MATRICES:
                    assert (
                        self.prepare_matrices == True
                    ), "this test only makes sense with prepare_matrices set to True"
                    poisson_mat = jnp.asarray(self.get_poisson_mat())
                    self.prepare_matrices = False
                    poisson_mat_ = jnp.asarray(self.get_poisson_mat())
                    jax.debug.print(
                        "poisson: {x}", x=jnp.linalg.norm(poisson_mat - poisson_mat_)
                    )
                    self.prepare_matrices = True

                phi_a_hat_new = lhs_mat_a_inv @ rhs_a
                v_1_lap_hat_new_a = phi_a_hat_new

                # compute velocity in y direction
                v_1_hat_new_a = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_a, jnp.asarray(self.get_poisson_mat()), kx, kz
                )
                v_1_hat_new_a = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_a, 1
                )

                v_1_hat_new_b = jnp.flip(v_1_hat_new_a)

                # reconstruct velocity s.t. hom. Neumann is fulfilled
                v_1_hat_new_p_diff = domain.diff_fourier_field_slice(
                    v_1_hat_new_p, 1, 1
                )
                v_1_hat_new_a_diff = domain.diff_fourier_field_slice(
                    v_1_hat_new_a, 1, 1
                )
                v_1_hat_new_b_diff = domain.diff_fourier_field_slice(
                    v_1_hat_new_b, 1, 1
                )
                M = jnp.array(
                    [
                        [v_1_hat_new_a_diff[0], v_1_hat_new_b_diff[0]],
                        [v_1_hat_new_a_diff[-1], v_1_hat_new_b_diff[-1]],
                    ]
                )
                R = jnp.array([-v_1_hat_new_p_diff[0], -v_1_hat_new_p_diff[-1]])
                a, b = jnp.linalg.lstsq(M, R)[0]
                v_1_hat_new = v_1_hat_new_p + a * v_1_hat_new_a + b * v_1_hat_new_b
                v_1_hat_new = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new, 1
                )

                # vorticity
                lhs_mat_vort_inv = jnp.asarray(self.get_rk_mats_lhs_inv(step, kx, kz))
                rhs_mat_vort = jnp.asarray(self.get_rk_mats_rhs(step, kx, kz))

                phi_vort_hat = vort_1_hat_sw

                N_vort_new = h_g_hat_sw
                N_vort_old = h_g_hat_old_sw

                rhs_vort = (
                    rhs_mat_vort @ phi_vort_hat
                    + (self.get_dt() * gamma[step]) * N_vort_new
                    + (self.get_dt() * xi[step]) * N_vort_old
                )

                rhs_vort = domain.update_boundary_conditions_fourier_field_slice(
                    rhs_vort, 1
                )

                phi_hat_vort_new = lhs_mat_vort_inv @ rhs_vort

                vort_1_hat_new = domain.update_boundary_conditions_fourier_field_slice(
                    phi_hat_vort_new, 1
                )

                # compute velocities in x and z directions
                def rk_00() -> Tuple["jnp_array", "jnp_array"]:
                    kx__ = 0
                    kz__ = 0
                    lhs_mat_inv_00 = jnp.asarray(self.get_rk_mats_lhs_inv_ns(step))
                    rhs_mat_00 = jnp.asarray(self.get_rk_mats_rhs_ns(step))

                    if TEST_MATRICES:
                        assert (
                            self.prepare_matrices == True
                        ), "this test only makes sense with prepare_matrices set to True"
                        self.prepare_matrices = False
                        lhs_mat_inv_00_ = jnp.asarray(self.get_rk_mats_lhs_inv_ns(step))
                        rhs_mat_00_ = jnp.asarray(self.get_rk_mats_rhs_ns(step))
                        jax.debug.print(
                            "ns lhs: {x}",
                            x=jnp.linalg.norm(lhs_mat_inv_00 - lhs_mat_inv_00_),
                        )
                        jax.debug.print(
                            "ns rhs: {x}", x=jnp.linalg.norm(rhs_mat_00 - rhs_mat_00_)
                        )
                        self.prepare_matrices = True

                    v_hat = jnp.block(
                        [
                            v_0_hat_sw_00,
                            v_2_hat_sw_00,
                        ]
                    )

                    dpdx = (
                        dPdx
                        * (
                            (
                                domain.get_shape_aliasing()[0]
                                * (2 * jnp.pi / domain.scale_factors[0]) ** 2
                            )
                            * (
                                domain.get_shape_aliasing()[2]
                                * (2 * jnp.pi / domain.scale_factors[2]) ** 2
                            )
                        )
                        ** 0.5
                    )

                    source_hat = self.get_source_term(time_step)
                    if type(source_hat) is NoneType:
                        source_x_00 = jnp.zeros_like(conv_ns_hat_sw_0_00)
                        source_z_00 = jnp.zeros_like(conv_ns_hat_sw_2_00)
                    else:
                        assert source_hat is not None
                        source_x_00 = source_hat[0][0, :, 0]
                        source_z_00 = source_hat[2][0, :, 0]

                    N_00_new = jnp.block(
                        [
                            -conv_ns_hat_sw_0_00,
                            -conv_ns_hat_sw_2_00,
                        ]
                    ) + jnp.block(
                        [
                            -dpdx * jnp.ones_like(conv_ns_hat_sw_0_00) + source_x_00,
                            0 * jnp.zeros_like(conv_ns_hat_sw_2_00) + source_z_00,
                        ]
                    )

                    N_00_old = jnp.block(
                        [
                            -conv_ns_hat_old_sw_0_00,
                            -conv_ns_hat_old_sw_2_00,
                        ]
                    ) + jnp.block(
                        [
                            -dpdx * jnp.ones_like(conv_ns_hat_sw_0_00) + source_x_00,
                            0 * jnp.zeros_like(conv_ns_hat_sw_2_00) + source_z_00,
                        ]
                    )
                    dpdx = None
                    v_hat_new = lhs_mat_inv_00 @ (
                        rhs_mat_00 @ v_hat
                        + (self.get_dt() * gamma[step]) * N_00_new
                        + (self.get_dt() * xi[step]) * N_00_old
                    )
                    return (v_hat_new[:n], v_hat_new[n:])

                def rk_not_00(kx: int, kz: int) -> Tuple["jnp_array", "jnp_array"]:
                    kx_ = jnp.asarray(domain.grid[0])[kx]
                    kz_ = jnp.asarray(domain.grid[2])[kz]
                    j_kx = 1j * kx_
                    j_kz = 1j * kz_
                    minus_kx_kz_sq = -(kx_**2 + kz_**2)
                    v_1_new_y = domain.diff_fourier_field_slice(v_1_hat_new, 1, 1)
                    # v_1_new_y = domain.update_boundary_conditions_fourier_field_slice(
                    #     v_1_new_y, 1
                    # )
                    # vort_1_hat_new_: "jnp_array" = (
                    #     domain.update_boundary_conditions_fourier_field_slice(
                    #         vort_1_hat_new, 1
                    #     )
                    # )
                    v_0_new = (
                        -j_kx * v_1_new_y + j_kz * vort_1_hat_new
                    ) / minus_kx_kz_sq
                    v_2_new = (
                        -j_kz * v_1_new_y - j_kx * vort_1_hat_new
                    ) / minus_kx_kz_sq
                    return (v_0_new, v_2_new)

                v_0_new_field, v_2_new_field = jax.lax.cond(
                    kx == 0,
                    lambda kx___, kz___: jax.lax.cond(
                        kz___ == 0,
                        lambda _, __: rk_00(),
                        lambda kx__, kz__: rk_not_00(kx__, kz__),
                        kx___,
                        kz___,
                    ),
                    lambda kx___, kz___: rk_not_00(kx___, kz___),
                    kx,
                    kz,
                )
                # if kx == 0 and kz == 0:
                #     v_0_new_field, v_2_new_field = rk_00()
                # else:
                #     v_0_new_field, v_2_new_field = rk_not_00(kx, kz)
                return (v_0_new_field, v_1_hat_new, v_2_new_field, v_1_lap_hat_new_a)

            return fn

        number_of_rk_steps = self.get_number_of_rk_steps()

        h_v_hat_old, h_g_hat_old, conv_ns_hat_old = (None, None, None)

        Nx = self.get_domain().number_of_cells(0)
        Ny = self.get_domain().number_of_cells(1)
        Nz = self.get_domain().number_of_cells(2)

        # @partial(
        #     jax.jit,
        #     static_argnums=(0,),
        #     # compiler_options={
        #     #     "exec_time_optimization_effort": 0.5,  # change here
        #     #     "memory_fitting_effort": 1.0,  # change here
        #     # },
        # )
        def get_new_vel_field_map(
            step: int,
            v_1_lap_hat: "jnp_array",
            vort_hat_1: "jnp_array",
            conv_ns_hat_0: "jnp_array",
            conv_ns_hat_2: "jnp_array",
            conv_ns_hat_old_0: "jnp_array",
            conv_ns_hat_old_2: "jnp_array",
            h_v_hat: "jnp_array",
            h_g_hat: "jnp_array",
            h_v_hat_old: "jnp_array",
            h_g_hat_old: "jnp_array",
        ) -> "jnp_array":
            number_of_input_arguments = 6

            conv_ns_hat_0_00 = conv_ns_hat_0[0, :, 0]
            conv_ns_hat_2_00 = conv_ns_hat_2[0, :, 0]
            conv_ns_hat_0_00_old = conv_ns_hat_old_0[0, :, 0]
            conv_ns_hat_2_00_old = conv_ns_hat_old_2[0, :, 0]
            v_0_hat_00 = vel_hat_data[0][0, :, 0]
            v_2_hat_00 = vel_hat_data[2][0, :, 0]

            def outer_map(
                kzs_: "np_jnp_array",
            ) -> Callable[["np_jnp_array"], "jnp_array"]:
                def fn(kx_state: "np_jnp_array") -> "jnp_array":
                    kx = kx_state[0]
                    fields_2d = jnp.split(
                        kx_state[1:], number_of_input_arguments, axis=0
                    )
                    for i in range(len(fields_2d)):
                        fields_2d[i] = jnp.reshape(fields_2d[i], (Nz, Ny)).T
                    state_slice = jnp.concatenate(fields_2d).T
                    kz_state_slice = jnp.concatenate([kzs_.T, state_slice], axis=1)
                    N = kz_state_slice.shape[0]
                    batch_size = N // 1
                    out: "jnp_array" = jax.lax.map(
                        inner_map(kx), kz_state_slice, batch_size=batch_size
                    )
                    # out = jax.vmap(inner_map(kx))(kz_state_slice)
                    return out

                return fn

            def inner_map(
                kx: "np_jnp_array",
            ) -> Callable[["np_jnp_array"], "jnp_array"]:
                def fn(kz_one_pt_state: "np_jnp_array") -> "jnp_array":
                    kz = kz_one_pt_state[0]
                    fields_1d = jnp.split(
                        kz_one_pt_state[1:],
                        number_of_input_arguments,
                        axis=0,
                    )
                    kx_int: int = kx.real.astype(int)  # type: ignore[assignment]
                    kz_int: int = kz.real.astype(int)  # type: ignore[assignment]
                    (
                        v_0_new_field,
                        v_1_hat_new,
                        v_2_new_field,
                        _,
                    ) = perform_single_rk_step_for_single_wavenumber(step)(
                        (kx_int, kz_int),
                        *fields_1d,
                        v_0_hat_00,
                        v_2_hat_00,
                        conv_ns_hat_0_00,
                        conv_ns_hat_2_00,
                        conv_ns_hat_0_00_old,
                        conv_ns_hat_2_00_old,
                    )  # type: ignore[call-arg]
                    return jnp.array([v_0_new_field, v_1_hat_new, v_2_new_field])

                return fn

            kx_arr = np.atleast_2d(np.arange(Nx))
            kz_arr = np.atleast_2d(np.arange(Nz))
            state = jnp.concatenate(
                [
                    jnp.moveaxis(v_1_lap_hat, 1, 2),
                    jnp.moveaxis(vort_hat_1, 1, 2),
                    jnp.moveaxis(h_v_hat, 1, 2),
                    jnp.moveaxis(h_g_hat, 1, 2),
                    jnp.moveaxis(h_v_hat_old, 1, 2),
                    jnp.moveaxis(h_g_hat_old, 1, 2),
                ],
                axis=1,
            )
            # state_ = jnp.reshape(state, (Nx, (number_of_input_arguments * Ny * Nz)))
            kx_state = jnp.concatenate(
                [
                    kx_arr.T,
                    jnp.reshape(state, (Nx, (number_of_input_arguments * Ny * Nz))),
                ],
                axis=1,
            )

            N = kx_state.shape[0]
            batch_size = N // 1
            out = jax.lax.map(outer_map(kz_arr), kx_state, batch_size=batch_size)
            # out = jax.vmap(outer_map(kz_arr))(kx_state)
            # return jnp.array([jnp.moveaxis(v, 1, 2) for v in out])
            out_ = jnp.moveaxis(out, 2, 0)
            return jnp.moveaxis(out_, 2, 3)

        for step in range(number_of_rk_steps):

            # filter out highest wavenumbers
            # vel_hat_data = jnp.array(
            #     [
            #         self.get_domain().filter_field_fourier_only(vel_hat_data[i])
            #         # self.get_domain().filter_field(vel_hat_data[i])
            #         for i in self.all_dimensions()
            #     ]
            # )
            # update nonlinear terms
            (
                h_v_hat,
                h_g_hat,
                vort_hat,
                conv_ns_hat,
            ) = self.nonlinear_update_fn(vel_hat_data, time_step)

            source_hat = self.get_source_term(time_step)
            if source_hat is not None:
                domain: FourierDomain = self.get_domain()
                curl_source = domain.curl(source_hat)[1]
                h_v_hat += (
                    domain.diff(curl_source[0], 2) - domain.diff(curl_source[2], 0)
                )[1]
                h_g_hat += curl_source[1]

            if step == 0:
                h_v_hat_old = jnp.zeros_like(h_v_hat)
                h_g_hat_old = jnp.zeros_like(h_g_hat)
                conv_ns_hat_old = jnp.zeros_like(conv_ns_hat)

            assert h_v_hat_old is not None
            assert h_g_hat_old is not None
            assert conv_ns_hat_old is not None

            # solve equations
            v_1_hat = vel_hat_data[1, ...]
            v_1_lap_hat = jnp.sum(
                jnp.array(
                    [
                        self.get_domain().diff(v_1_hat, i, 2)
                        for i in self.all_dimensions()
                    ]
                ),
                axis=0,
            )

            vel_new_hat_field = get_new_vel_field_map(
                step,
                v_1_lap_hat,
                vort_hat[1],
                conv_ns_hat[0],
                conv_ns_hat[2],
                conv_ns_hat_old[0],
                conv_ns_hat_old[2],
                h_v_hat,
                h_g_hat,
                h_v_hat_old,
                h_g_hat_old,
            )

            h_v_hat_old, h_g_hat_old, conv_ns_hat_old = (
                h_v_hat,
                h_g_hat,
                conv_ns_hat,
            )

            vel_new_hat_field = jnp.array(
                [
                    self.get_domain().update_boundary_conditions(vel_new_hat_field[i])
                    for i in self.all_dimensions()
                ]
            )

            vel_hat_data = vel_new_hat_field

        vel_hat_data, dPdx = self.enforce_constant_mass_flux(
            vel_hat_data, dPdx, time_step
        )

        if not Field.activate_jit_:
            vel_new_hat = VectorField(
                [
                    FourierField(
                        self.get_physical_domain(),
                        vel_hat_data[i],
                        name="velocity_hat_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            vel_new_hat.name = "velocity_hat"
            for i in self.all_dimensions():
                vel_new_hat[i].name = "velocity_hat_" + "xyz"[i]
            self.append_field("velocity_hat", vel_new_hat, in_place=False)
        return vel_hat_data, dPdx

    # @partial(jax.jit, static_argnums=(0, 2))
    def perform_time_step(
        self,
        vel_hat_data: Optional["jnp_array"] = None,
        dPdx: Optional["float"] = None,
        time_step: Optional[int] = None,
    ) -> Tuple["jnp_array", "jsd_float"]:
        if type(vel_hat_data) == NoneType:
            vel_hat_data_ = self.get_latest_field("velocity_hat").get_data()
        else:
            assert vel_hat_data is not None
            vel_hat_data_ = vel_hat_data
        if type(dPdx) == NoneType:
            dPdx_ = self.dPdx
        else:
            dPdx_ = dPdx

        assert time_step is not None
        vel_hat_data_new_, dPdx_ = self.perform_runge_kutta_step(
            vel_hat_data_, dPdx_, time_step
        )
        if type(vel_hat_data) == NoneType:
            vel_hat_new = VectorField(
                [
                    FourierField(
                        self.get_physical_domain(),
                        vel_hat_data_new_[i],
                        name="velocity_hat_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            vel_hat_new.set_time_step(
                self.get_latest_field("velocity_hat").get_time_step()
            )
            self.append_field("velocity_hat", vel_hat_new)
        return vel_hat_data_new_, dPdx_

    def __hash__(self):
        return self.get_initial_field("velocity_hat").no_hat().energy()

    def __eq__(self, other):

        return isinstance(other, Equation) and (
            self.get_initial_field("velocity_hat").no_hat().energy()
            == other.get_initial_field("velocity_hat").no_hat().energy()
        )

    @partial(jax.jit, static_argnums=(0))
    def solve_scan(
        self,
    ) -> Tuple["jnp_array", "List[jsd_float]", int]:
        cfl_initial = self.get_cfl()
        print_verb("initial cfl:", cfl_initial, debug=True, verbosity_level=2)

        def inner_step_fn(
            u0: Tuple["jnp_array", "jsd_float", int], _: Any
        ) -> Tuple[Tuple["jnp_array", "jsd_float", int], None]:
            u0_, dPdx, time_step = u0
            out, dPdx = self.perform_time_step(u0_, cast(float, dPdx), time_step)
            return ((out, dPdx, time_step + 1), None)

        def step_fn(
            u0: Tuple["jnp_array", "jsd_float", int], _: Any
        ) -> Tuple[
            Tuple["jnp_array", "jsd_float", int], Tuple["jnp_array", "jsd_float", int]
        ]:
            out, _ = jax.lax.scan(
                jax.checkpoint(inner_step_fn),  # type: ignore[attr-defined]
                u0,
                xs=None,
                length=number_of_inner_steps,
                # inner_step_fn, u0, xs=None, length=number_of_inner_steps
            )
            return out, out

        def median_factor(n: int) -> int:
            """Return the median integer factor of n."""
            factors = reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
            factors.sort()
            number_of_factors = len(factors)  # should always be divisible by 2
            return factors[number_of_factors // 2]

        u0 = self.get_initial_field("velocity_hat").get_data()
        dPdx = self.dPdx
        self.dPdx = 0.0
        ts = jnp.arange(0, self.end_time, self.get_dt())
        number_of_time_steps = len(ts)

        vb = 2
        if not self.write_entire_output:
            number_of_inner_steps = median_factor(number_of_time_steps)
            number_of_outer_steps = number_of_time_steps // number_of_inner_steps
            # number_of_outer_steps = median_factor(number_of_time_steps)
            # number_of_inner_steps = number_of_time_steps // number_of_outer_steps
            vb = 2
            if (
                abs(np.sqrt(number_of_time_steps)) - number_of_outer_steps
                > number_of_outer_steps
            ):
                print_verb(
                    "WARNING: bad division into inner/outer steps detected. Consider adjusting your time step size and/or your final time to allow for a number of time steps with more divisors."
                )
                vb = 1
        else:
            number_of_outer_steps = number_of_time_steps
            number_of_inner_steps = 1

        self.number_of_time_steps = number_of_time_steps
        self.number_of_outer_steps = number_of_outer_steps
        self.number_of_inner_steps = number_of_inner_steps

        start_step = self.get_initial_field("velocity_hat").get_time_step()

        print_verb(
            "Dividing "
            + str(number_of_time_steps)
            + " time steps into "
            + str(number_of_inner_steps)
            + " inner steps and "
            + str(number_of_outer_steps)
            + " outer steps.",
            verbosity_level=vb,
        )

        if self.write_intermediate_output and not self.write_entire_output:
            u_final, trajectory = jax.lax.scan(
                jax.checkpoint(step_fn), (u0, dPdx, 0), xs=None, length=number_of_outer_steps  # type: ignore
            )
            t = 0
            # for u in trajectory[0]:
            #     velocity = VectorField(
            #         [
            #             FourierField(
            #                 self.get_physical_domain(),
            #                 u[i],
            #                 name="velocity_hat_" + "xyz"[i],
            #             )
            #             for i in self.all_dimensions()
            #         ]
            #     )
            #     velocity.set_time_step(start_step + t)
            #     t += 1
            # self.append_field("velocity_hat", velocity, in_place=False)
            # if Equation.verbosity_level >= 3:
            #     for i in range(self.get_number_of_fields("velocity_hat")):
            #         cfl_s = self.get_cfl(i)
            #         print_verb("i: ", i, "cfl:", cfl_s, verbosity_level=3)
            # cfl_final = self.get_cfl()
            # print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
            out = jnp.insert(
                trajectory[0],
                0,
                self.get_initial_field("velocity_hat").get_data(),
                axis=0,
            )
            return (out, cast("List[jsd_float]", trajectory[1]), len(ts))
        elif self.write_entire_output:
            u_final, trajectory = jax.lax.scan(
                step_fn, (u0, dPdx, 0), xs=None, length=number_of_outer_steps
            )
            # velocity_final = VectorField(
            #     [
            #         FourierField(
            #             self.get_physical_domain(),
            #             trajectory[0][-1][i],
            #             name="velocity_hat_" + "xyz"[i],
            #         )
            #         for i in self.all_dimensions()
            #     ]
            # )
            # velocity_final.set_time_step(start_step + number_of_outer_steps)
            # self.append_field("velocity_hat", velocity_final, in_place=False)
            # cfl_final = self.get_cfl()
            # print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
            out = jnp.insert(
                trajectory[0],
                0,
                self.get_initial_field("velocity_hat").get_data(),
                axis=0,
            )
            return (out, cast("List[jsd_float]", trajectory[1]), len(ts))
        else:
            u_final, _ = jax.lax.scan(
                step_fn, (u0, dPdx, 0), xs=None, length=number_of_outer_steps
            )
            # velocity_final = VectorField(
            #     [
            #         FourierField(
            #             self.get_physical_domain(),
            #             u_final[0][i],
            #             name="velocity_hat_" + "xyz"[i],
            #         )
            #         for i in self.all_dimensions()
            #     ]
            # )
            # velocity_final.set_time_step(start_step + number_of_outer_steps)
            # # self.append_field("velocity_hat", velocity_final, in_place=False)
            cfl_final = self.get_cfl()
            print_verb("final cfl:", cfl_final, debug=True, verbosity_level=2)
            # return (velocity_final, [u_final[1]], len(ts))
            return (u_final[0], [u_final[1]], len(ts))

    def post_process(self: E) -> None:
        if type(self.post_process_fn) != NoneType:
            assert self.post_process_fn is not None
            for i in range(self.get_number_of_fields("velocity_hat")):
                self.post_process_fn(self, i)


def solve_navier_stokes_laminar(
    Re: float = 1.8e2,
    end_time: float = 1e1,
    Nx: int = 8,
    Ny: int = 40,
    Nz: int = 8,
    perturbation_factor: float = 0.1,
    scale_factors: Tuple[float, float, float] = (1.87, 1.0, 0.93),
    aliasing: float = 1.0,
    **params: Any,
) -> NavierStokesVelVort:

    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )
    cmf = params.get("constant_mass_flux", False)
    if cmf:
        u_max_over_u_tau = params.get("u_max_over_u_tau", 1.0)
    else:
        u_max_over_u_tau = params.get("u_max_over_u_tau", Re / 2.0)

    vel_x_fn_ana: "Vel_fn_type" = (
        lambda X: u_max_over_u_tau * (1 - X[1] ** 2) + 0.0 * X[0] * X[2]
    )

    vel_x_fn: "Vel_fn_type" = lambda X: jnp.pi / 3 * u_max_over_u_tau * (
        perturbation_factor
        * jnp.cos(X[1] * jnp.pi / 2)
        * (jnp.cos(3 * X[0]) ** 2 * jnp.cos(4 * X[2]) ** 2)
    ) + (1 - perturbation_factor) * vel_x_fn_ana(X)

    # add small perturbation in y and z to see if it decays
    vel_y_fn: "Vel_fn_type" = (
        lambda X: 0.1
        * perturbation_factor
        * u_max_over_u_tau
        * (
            jnp.pi
            / 3
            # * jnp.cos(X[1] * jnp.pi / 2)
            * (1 - X[1] ** 2) ** 2
            * jnp.cos(3 * X[0])
            * jnp.cos(4 * X[2])
        )
    )
    vel_z_fn: "Vel_fn_type" = (
        lambda X: 0.1
        * jnp.pi
        / 3
        * perturbation_factor
        * u_max_over_u_tau
        * (jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(5 * X[0]) * jnp.cos(3 * X[2]))
    )
    vel_x = PhysicalField.FromFunc(domain, vel_x_fn, name="velocity_x")
    vel_y = PhysicalField.FromFunc(domain, vel_y_fn, name="velocity_y")
    vel_z = PhysicalField.FromFunc(domain, vel_z_fn, name="velocity_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")

    nse = NavierStokesVelVort.FromVelocityField(vel, Re=Re, **params)
    nse.end_time = end_time

    nse.before_time_step_fn = None
    nse.after_time_step_fn = None

    def post_process(nse_: NavierStokesVelVort, i: int) -> None:
        n_steps = nse_.get_number_of_fields("velocity_hat")
        vel_hat: VectorField[FourierField] = nse_.get_field("velocity_hat", i)
        vel: VectorField[PhysicalField] = vel_hat.no_hat()

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
            vel_hat_: VectorField[FourierField] = nse_.get_field("velocity_hat", j)
            vel_: VectorField[PhysicalField] = vel_hat_.no_hat()
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
        gain = energy_t[-1] / energy_t[0]

    nse.set_post_process_fn(post_process)

    return nse

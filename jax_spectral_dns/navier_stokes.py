#!/usr/bin/env python3
from __future__ import annotations

NoneType = type(None)
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


def update_nonlinear_terms_high_performance(
    physical_domain: PhysicalDomain,
    fourier_domain: FourierDomain,
    vel_hat_new: "jnp_array",
) -> Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]:

    vort_hat_new = fourier_domain.curl(vel_hat_new)
    vel_new = jnp.array(
        [
            fourier_domain.filter_field_nonfourier_only(
                fourier_domain.field_no_hat(vel_hat_new.at[i].get())
            )
            # fourier_domain.field_no_hat(vel_hat_new.at[i].get())
            for i in jnp.arange(physical_domain.number_of_dimensions)
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


class NavierStokesVelVort(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation)"

    def __init__(self, velocity_field: VectorField[FourierField], **params: Any):
        if "physical_domain" in params:
            physical_domain = params["physical_domain"]
            domain = physical_domain.hat()
        else:
            domain = velocity_field[0].fourier_domain
            physical_domain = velocity_field[0].physical_domain

        u_max_over_u_tau = params.get("u_max_over_u_tau", 1.0)

        dt = params.get("dt", 1e-2)

        max_cfl = params.get("max_cfl", 0.7)

        try:
            Re_tau = params["Re_tau"]
        except KeyError:
            try:
                Re_tau = params["Re"] / u_max_over_u_tau
            except KeyError:
                raise Exception("Either Re or Re_tau has to be given as a parameter.")
        self.nonlinear_update_fn: Callable[
            ["jnp_array"], Tuple["jnp_array", "jnp_array", "jnp_array", "jnp_array"]
        ] = lambda vel: update_nonlinear_terms_high_performance(
            self.get_physical_domain(), self.get_domain(), vel
        )
        super().__init__(domain, velocity_field, dt=dt)

        poisson_mat = domain.assemble_poisson_matrix()
        (
            rk_mats_rhs,
            rk_mats_lhs_inv,
            rk_rhs_inhom,
            rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns,
        ) = self.prepare_assemble_rk_matrices(domain, physical_domain, Re_tau, dt)

        poisson_mat.setflags(write=False)
        rk_mats_rhs.setflags(write=False)
        rk_mats_lhs_inv.setflags(write=False)
        rk_rhs_inhom.setflags(write=False)
        rk_mats_lhs_inv_inhom.setflags(write=False)
        rk_mats_rhs_ns.setflags(write=False)
        rk_mats_lhs_inv_ns.setflags(write=False)

        self.nse_fixed_parameters = NavierStokesVelVortFixedParameters(
            physical_domain,
            poisson_mat,
            rk_mats_rhs,
            rk_mats_lhs_inv,
            rk_rhs_inhom,
            rk_mats_lhs_inv_inhom,
            rk_mats_rhs_ns,
            rk_mats_lhs_inv_ns,
            Re_tau,
            max_cfl,
            u_max_over_u_tau,
        )
        self.update_flow_rate()
        print_verb("calculated flow rate: ", self.flow_rate, verbosity_level=3)

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

    def get_poisson_mat(self) -> "np_complex_array":
        return self.nse_fixed_parameters.poisson_mat

    def get_rk_mats_lhs_inv(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_mats_lhs_inv

    def get_rk_mats_rhs(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_mats_rhs

    def get_rk_mats_lhs_inv_inhom(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_mats_lhs_inv_inhom

    def get_rk_rhs_inhom(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_rhs_inhom

    def get_rk_mats_lhs_inv_ns(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_mats_lhs_inv_ns

    def get_rk_mats_rhs_ns(self) -> "np_complex_array":
        return self.nse_fixed_parameters.rk_mats_rhs_ns

    def get_Re_tau(self) -> "jsd_float":
        return self.nse_fixed_parameters.Re_tau

    def get_max_cfl(self) -> "jsd_float":
        return self.nse_fixed_parameters.max_cfl

    def get_dt_update_frequency(self) -> "jsd_float":
        return self.nse_fixed_parameters.dt_update_frequency

    def get_u_max_over_u_tau(self) -> "jsd_float":
        return self.nse_fixed_parameters.u_max_over_u_tau

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

        hel_hat = velocity_field_hat.cross_product(vort_hat)
        for i in jnp.arange(3):
            hel_hat[i].name = "hel_hat_" + str(i)
        return (vort_hat, hel_hat)

    def get_flow_rate(self) -> "jsd_float":
        vel_hat: VectorField[FourierField] = self.get_latest_field("velocity_hat")
        vel_hat_0: FourierField = vel_hat[0]
        int: PhysicalField = vel_hat_0.no_hat().definite_integral(1)  # type: ignore[assignment]
        return cast("jsd_float", int[0, 0])

    def update_flow_rate(self) -> None:
        self.flow_rate = self.get_flow_rate()
        dPdx = -self.flow_rate * 3 / 2 / self.get_Re_tau()
        self.dpdx = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: dPdx + 0.0 * X[0] * X[1] * X[2]
        ).hat()
        self.dpdz = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: 0.0 + 0.0 * X[0] * X[1] * X[2]
        ).hat()

    def get_cheb_mat_2_homogeneous_dirichlet(self) -> "np_float_array":
        return self.get_initial_field("velocity_hat")[
            0
        ].get_cheb_mat_2_homogeneous_dirichlet(1)

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

    def get_rk_parameters(self) -> Tuple[List["jsd_float"], ...]:
        return (
            [29 / 96, -3 / 40, 1 / 6],
            [37 / 160, 5 / 24, 1 / 6],
            [8 / 15, 5 / 12, 3 / 4],
            [0, -17 / 60, -5 / 12],
        )

    def prepare_assemble_rk_matrices(
        self,
        domain: FourierDomain,
        physical_domain: PhysicalDomain,
        Re_tau: "jsd_float",
        dt: "jsd_float",
    ) -> Tuple["np_complex_array", ...]:
        alpha, beta, _, _ = self.get_rk_parameters()
        D2 = np.linalg.matrix_power(physical_domain.diff_mats[1], 2)
        Ly = 1 / Re_tau * D2
        n = Ly.shape[0]
        I = np.eye(n)
        Z = np.zeros((n, n))
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        L_NS_y = 1 / Re_tau * np.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])
        rk_mats_rhs = np.zeros(
            (3, domain.number_of_cells(0), domain.number_of_cells(2), n, n),
            dtype=np.complex128,
        )
        rk_mats_lhs_inv = np.zeros(
            (3, domain.number_of_cells(0), domain.number_of_cells(2), n, n),
            dtype=np.complex128,
        )
        rk_rhs_inhom = np.zeros(
            (3, domain.number_of_cells(0), domain.number_of_cells(2), n),
            dtype=np.complex128,
        )
        rk_mats_lhs_inv_inhom = np.zeros(
            (3, domain.number_of_cells(0), domain.number_of_cells(2), n, n),
            dtype=np.complex128,
        )
        rk_mats_rhs_ns = np.zeros((3, 2 * n, 2 * n), dtype=np.complex128)
        rk_mats_lhs_inv_ns = np.zeros((3, 2 * n, 2 * n), dtype=np.complex128)
        for i in range(3):
            for xi, kx in enumerate(domain.grid[0]):
                for zi, kz in enumerate(domain.grid[2]):
                    L = Ly + I * (-(kx**2 + kz**2)) / Re_tau
                    rhs_mat = I + alpha[i] * dt * L
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
            lhs_mat_ns = I_ns - beta[i] * dt * L_ns
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
                vel_y[0, :, 0],
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
            vel_1_y_ = domain.diff_fourier_field_slice(vel_y_, 1, 1)
            # vel_1_y_ = domain.update_boundary_conditions_fourier_field_slice(
            #     vel_1_y_, 1
            # )
            vel_x_ = (-j_kx * vel_1_y_ + j_kz * vort_) / minus_kx_kz_sq
            if two_d:
                vel_z_ = jnp.zeros_like(vel_x_, dtype=jnp.complex128)
            else:
                vel_z_ = (-j_kz * vel_1_y_ - j_kx * vort_) / minus_kx_kz_sq

            # vel_y_ = domain.integrate_fourier_field_slice(vel_1_y_, 1, 1, bc_right=0.0, bc_left=0.0)
            # vel_y_ = domain.update_boundary_conditions_fourier_field_slice(
            #     vel_y_, 1
            # )
            return (
                vel_x_.astype(jnp.complex128),
                vel_y_.astype(jnp.complex128),
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
                out: "jnp_array" = jax.lax.map(inner_map(kx), kz_state_slice)  # type: ignore[no-untyped-call]
                return out

            return fn

        vel_y = domain.update_boundary_conditions(vel_y)

        vort = domain.update_boundary_conditions(vort)

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
        out = jax.lax.map(outer_map(kz_arr), kx_state)  # type: ignore[no-untyped-call]
        u_w = [jnp.moveaxis(v, 1, 2) for v in out]
        return jnp.array([u_w[0], u_w[1], u_w[2]])

    def perform_runge_kutta_step(self, vel_hat_data: "jnp_array") -> "jnp_array":

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
                Optional["jnp_array"],
                Optional["jnp_array"],
                "jnp_array",
                "jnp_array",
                "jnp_array",
                "jnp_array",
                Optional["jnp_array"],
                Optional["jnp_array"],
            ],
            Tuple["jnp_array", ...],
        ]:
            def fn(
                K: Tuple[int, int],
                v_1_lap_hat_sw: "jnp_array",
                vort_1_hat_sw: "jnp_array",
                h_v_hat_sw: "jnp_array",
                h_g_hat_sw: "jnp_array",
                h_v_hat_old_sw: Optional["jnp_array"],
                h_g_hat_old_sw: Optional["jnp_array"],
                v_0_hat_sw_00: "jnp_array",
                v_2_hat_sw_00: "jnp_array",
                conv_ns_hat_sw_0_00: "jnp_array",
                conv_ns_hat_sw_2_00: "jnp_array",
                conv_ns_hat_old_sw_0_00: Optional["jnp_array"],
                conv_ns_hat_old_sw_2_00: Optional["jnp_array"],
            ) -> Tuple["jnp_array", ...]:
                domain = self.get_domain()
                kx = K[0]
                kz = K[1]
                # kx_ = domain.grid[0][kx]
                # kz_ = domain.grid[2][kz]

                # wall-normal velocity
                # p-part
                # L_p_y = 1 / Re * D2
                # lhs_mat_p_, rhs_mat_p_ = self.assemble_rk_matrices(L_p_y, kx_, kz_, step)
                # lhs_mat_p_ = domain.enforce_homogeneous_dirichlet(lhs_mat_p_)
                lhs_mat_p_inv = jnp.asarray(self.get_rk_mats_lhs_inv())[step, kx, kz]
                rhs_mat_p = jnp.asarray(self.get_rk_mats_rhs())[step, kx, kz]
                # jax.debug.print("{x}", x = np.linalg.norm(rhs_mat_p - rhs_mat_p_))
                # jax.debug.print("{x}", x = np.linalg.norm(lhs_mat_p_inv - np.linalg.inv(lhs_mat_p_)))

                phi_hat_lap = v_1_lap_hat_sw

                N_p_new = h_v_hat_sw
                if type(h_v_hat_old_sw) == NoneType:
                    N_p_old: "jnp_array" = N_p_new
                else:
                    N_p_old = h_v_hat_old_sw  # type: ignore[assignment]

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
                lhs_mat_a_inv = jnp.asarray(self.get_rk_mats_lhs_inv_inhom())[
                    step, kx, kz
                ]
                rhs_a = jnp.asarray(self.get_rk_rhs_inhom())[step, kx, kz]
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
                lhs_mat_vort_inv = jnp.asarray(self.get_rk_mats_lhs_inv())[step, kx, kz]
                rhs_mat_vort = jnp.asarray(self.get_rk_mats_rhs())[step, kx, kz]

                phi_vort_hat = vort_1_hat_sw

                N_vort_new = h_g_hat_sw
                if type(h_g_hat_old_sw == NoneType):
                    N_vort_old = N_vort_new
                else:
                    N_vort_old = h_g_hat_old_sw  # type: ignore[assignment]

                rhs_vort = (
                    rhs_mat_vort @ phi_vort_hat
                    + (self.get_dt() * gamma[step]) * N_vort_new
                    + (self.get_dt() * xi[step]) * N_vort_old
                )

                rhs_vort = domain.update_boundary_conditions_fourier_field_slice(
                    rhs_vort, 1
                )

                phi_hat_vort_new = lhs_mat_vort_inv @ (rhs_vort)

                vort_1_hat_new = domain.update_boundary_conditions_fourier_field_slice(
                    phi_hat_vort_new, 1
                )

                # compute velocities in x and z directions
                def rk_00() -> Tuple["jnp_array", "jnp_array"]:
                    kx__ = 0
                    kz__ = 0
                    lhs_mat_inv_00 = jnp.asarray(self.get_rk_mats_lhs_inv_ns())[step]
                    rhs_mat_00 = jnp.asarray(self.get_rk_mats_rhs_ns())[step]

                    v_hat = jnp.block(
                        [
                            v_0_hat_sw_00,
                            v_2_hat_sw_00,
                        ]
                    )
                    N_00_new = jnp.block(
                        [
                            -conv_ns_hat_sw_0_00,
                            -conv_ns_hat_sw_2_00,
                        ]
                    ) + jnp.block(
                        [
                            -self.dpdx[kx__, :, kz__],
                            -self.dpdz[kx__, :, kz__],
                        ]
                    )

                    if type(conv_ns_hat_old == NoneType):
                        N_00_old = N_00_new
                    else:
                        assert conv_ns_hat_old_sw_0_00 is not None
                        assert conv_ns_hat_old_sw_2_00 is not None
                        N_00_old = jnp.block(
                            [
                                -conv_ns_hat_old_sw_0_00,
                                -conv_ns_hat_old_sw_2_00,
                            ]
                        ) + jnp.block(
                            [
                                -self.dpdx[kx__, :, kz__],
                                -self.dpdz[kx__, :, kz__],
                            ]
                        )
                    assert N_00_old is not None
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
                    v_1_new_y = domain.update_boundary_conditions_fourier_field_slice(
                        v_1_new_y, 1
                    )
                    vort_1_hat_new_: "jnp_array" = (
                        domain.update_boundary_conditions_fourier_field_slice(
                            vort_1_hat_new, 1
                        )
                    )
                    v_0_new = (
                        -j_kx * v_1_new_y + j_kz * vort_1_hat_new_
                    ) / minus_kx_kz_sq
                    v_2_new = (
                        -j_kz * v_1_new_y - j_kx * vort_1_hat_new_
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

        number_of_rk_steps = 3

        h_v_hat_old, h_g_hat_old, conv_ns_hat_old = (None, None, None)

        for step in range(number_of_rk_steps):

            # filter out high wavenumbers to dealias
            vel_hat_data = jnp.array(
                [
                    self.get_domain().filter_field_fourier_only(vel_hat_data[i])
                    # self.get_domain().filter_field(vel_hat_data[i])
                    for i in self.all_dimensions()
                ]
            )
            # update nonlinear terms
            (
                h_v_hat,
                h_g_hat,
                vort_hat,
                conv_ns_hat,
            ) = self.nonlinear_update_fn(vel_hat_data)

            if type(h_v_hat_old) == NoneType:
                h_v_hat_old = h_v_hat
            if type(h_g_hat_old) == NoneType:
                h_g_hat_old = h_g_hat
            if type(conv_ns_hat_old) == NoneType:
                conv_ns_hat_old = conv_ns_hat

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

            def get_new_vel_field_map(
                Nx: int,
                Ny: int,
                Nz: int,
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
                        out: "jnp_array" = jax.lax.map(inner_map(kx), kz_state_slice)  # type: ignore[no-untyped-call]
                        return out

                    return fn

                def inner_map(
                    kx: "np_jnp_array",
                ) -> Callable[["np_jnp_array"], List["jnp_array"]]:
                    def fn(kz_one_pt_state: "np_jnp_array") -> List["jnp_array"]:
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
                        return [v_0_new_field, v_1_hat_new, v_2_new_field]

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

                out = jax.lax.map(outer_map(kz_arr), kx_state)  # type: ignore[no-untyped-call]
                return jnp.array([jnp.moveaxis(v, 1, 2) for v in out])

            Nx = self.get_domain().number_of_cells(0)
            Ny = self.get_domain().number_of_cells(1)
            Nz = self.get_domain().number_of_cells(2)

            vel_new_hat_field = get_new_vel_field_map(
                Nx,
                Ny,
                Nz,
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

        if not Field.activate_jit_:
            vel_new_hat = VectorField(
                [
                    FourierField(
                        self.get_physical_domain(),
                        vel_new_hat_field[i],
                        name="velocity_hat_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            vel_new_hat.name = "velocity_hat"
            for i in self.all_dimensions():
                vel_new_hat[i].name = "velocity_hat_" + "xyz"[i]
            self.append_field("velocity_hat", vel_new_hat, in_place=False)
        # return vel_new_hat.get_data()
        return vel_new_hat_field

    def perform_time_step(
        self, vel_hat_data: Optional["jnp_array"] = None
    ) -> "jnp_array":
        if type(vel_hat_data) == NoneType:
            vel_hat_data_ = self.get_latest_field("velocity_hat").get_data()
        else:
            assert vel_hat_data is not None
            vel_hat_data_ = vel_hat_data
        vel_hat_data_new_ = self.perform_runge_kutta_step(vel_hat_data_)
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
            self.append_field("velocity_hat", vel_hat_new)
        return vel_hat_data_new_

    def solve_scan(self) -> Tuple[VectorField[FourierField], int]:
        cfl_initial = self.get_cfl()
        print_verb("initial cfl:", cfl_initial, debug=True)

        def inner_step_fn(u0: "jnp_array", _: Any) -> Tuple["jnp_array", None]:
            out = self.perform_time_step(u0)
            return (out, None)

        def step_fn(u0: "jnp_array", _: Any) -> Tuple["jnp_array", "jnp_array"]:
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

        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec
        from jax.sharding import NamedSharding
        from jax.experimental import mesh_utils

        P = jax.sharding.PartitionSpec
        n = jax.local_device_count()
        devices = mesh_utils.create_device_mesh((n,))
        mesh = jax.sharding.Mesh(devices, ("x",))
        sharding = jax.sharding.NamedSharding(mesh, P("x"))  # type: ignore[no-untyped-call]
        u0 = jax.device_put(self.get_latest_field("velocity_hat").get_data(), sharding)
        ts = jnp.arange(0, self.end_time, self.get_dt())
        number_of_time_steps = len(ts)

        number_of_inner_steps = median_factor(number_of_time_steps)
        number_of_outer_steps = number_of_time_steps // number_of_inner_steps

        vb = 2
        if (
            abs(np.sqrt(number_of_time_steps)) - number_of_outer_steps
            > number_of_outer_steps
        ):
            print_verb(
                "WARNING: bad division into inner/outer steps detected. Consider adjusting your time step size and/or your final time to allow for a number of time steps with more divisors."
            )
            vb = 1

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
        assert (
            number_of_inner_steps >= number_of_outer_steps
        ), "Something went wrong with inner/outer step division."
        if self.write_intermediate_output:
            u_final, trajectory = jax.lax.scan(
                step_fn, u0, xs=None, length=number_of_outer_steps
            )
            for u in trajectory:
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
            for i in range(self.get_number_of_fields("velocity_hat")):
                cfl_s = self.get_cfl(i)
                print_verb("i: ", i, "cfl:", cfl_s)
            return (velocity, len(ts))
        else:
            u_final, _ = jax.lax.scan(
                step_fn, u0, xs=None, length=number_of_outer_steps
            )
            velocity_final = VectorField(
                [
                    FourierField(
                        self.get_physical_domain(),
                        u_final[i],
                        name="velocity_hat_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            self.append_field("velocity_hat", velocity_final, in_place=False)
            cfl_final = self.get_cfl()
            print_verb("final cfl:", cfl_final, debug=True)
            return (velocity_final, len(ts))

    def post_process(self: E) -> None:
        if type(self.post_process_fn) != NoneType:
            assert self.post_process_fn is not None
            for i in range(self.get_number_of_fields("velocity_hat")):
                self.post_process_fn(self, i)


def solve_navier_stokes_laminar(
    Re: float = 1.8e2,
    end_time: float = 1e1,
    max_iter: int = 10000,
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
    # domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))
    u_max_over_u_tau = 1.0

    vel_x_fn_ana: "Vel_fn_type" = (
        lambda X: -1 * u_max_over_u_tau * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
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
    nse.max_iter = max_iter

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

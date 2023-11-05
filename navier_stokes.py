#!/usr/bin/env python3

NoneType = type(None)
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from functools import partial

from importlib import reload
import sys

from numpy import vectorize

try:
    reload(sys.modules["domain"])
except:
    if hasattr(sys, 'ps1'):
        pass
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    if hasattr(sys, 'ps1'):
        pass
from field import Field, VectorField, FourierField, FourierFieldSlice

try:
    reload(sys.modules["equation"])
except:
    if hasattr(sys, 'ps1'):
        pass
from equation import Equation

try:
    reload(sys.modules["linear_stability_calculation"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load linear stability")
from linear_stability_calculation import LinearStabilityCalculation

def update_nonlinear_terms_high_performance(domain, vel_hat_new):
    vel_new = jnp.array(
        [
            domain.no_hat(vel_hat_new.at[i].get())
            for i in jnp.arange(domain.number_of_dimensions)
        ]
    )
    vort_new = domain.curl(vel_new)

    vel_new_sq = 0
    for j in domain.all_dimensions():
        vel_new_sq += vel_new[j] * vel_new[j]
    vel_new_sq_nabla = []
    for i in domain.all_dimensions():
        vel_new_sq_nabla.append(domain.diff(vel_new_sq, i))

    hel_new = jnp.array(domain.cross_product(vel_new, vort_new)) - 1 / 2 * jnp.array(
        vel_new_sq_nabla
    )

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


class NavierStokesVelVort(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation)"
    max_cfl = 0.1
    # max_cfl = 0.7/10
    # max_cfl = 5e-2
    # max_dt = 1e10
    max_dt = 3e-3
    # u_max_over_u_tau = 1e2
    u_max_over_u_tau = 1e0

    def __init__(self, velocity_field, base_field=None, **params):
        domain = velocity_field[0].domain
        self.domain_no_hat = velocity_field[0].domain_no_hat

        try:
            self.Re_tau = params["Re_tau"]
        except KeyError:
            try:
                self.Re_tau = params["Re"] / self.u_max_over_u_tau
            except KeyError:
                raise Exception("Either Re or Re_tau has to be given as a parameter.")
        self.nonlinear_update_fn = update_nonlinear_terms_high_performance
        (
            h_v_hat_field,
            h_g_hat_field,
            vort_hat_field,
            conv_ns_hat_field,
        ) = self.update_nonlinear_terms(velocity_field)
        self.poisson_mat = None
        self.lhs_mat_inv = []
        self.rhs_mat = []
        if type(base_field) == NoneType:
            super().__init__(
                domain,
                velocity_field,
                h_v_hat_field,
                h_g_hat_field,
                vort_hat_field,
                conv_ns_hat_field,
                **params
            )
        else:
            super().__init__(
                domain,
                velocity_field,
                h_v_hat_field,
                h_g_hat_field,
                vort_hat_field,
                conv_ns_hat_field,
                base_field,
                **params
            )
        self.dt = self.get_time_step()
        self.update_flow_rate()
        print("calculated flow rate: ", self.flow_rate)

    @classmethod
    def FromVelocityField(cls, velocity_field, Re=1.8e2, end_time=1e0):
        velocity_field_hat = velocity_field.hat()
        velocity_field_hat.name = "velocity_hat"
        return cls(velocity_field_hat, Re=Re, end_time=end_time)

    @classmethod
    def FromRandom(cls, shape, Re, end_time=1e0):
        domain = Domain(shape, (True, False, True))
        vel_x = Field.FromRandom(domain, name="u0")
        vel_y = Field.FromRandom(domain, name="u1")
        vel_z = Field.FromRandom(domain, name="u2")
        # vel_y.update_boundary_conditions()
        vel = VectorField([vel_x, vel_y, vel_z], "velocity")
        return cls.FromVelocityField(vel, Re, end_time=end_time)

    def init_velocity(self, velocity_hat):
        self.set_field("velocity_hat", 0, velocity_hat)

    def get_vorticity_and_helicity(self):
        velocity_field_hat = self.get_latest_field("velocity_hat")
        vort_hat = velocity_field_hat.curl()
        for i in jnp.arange(3):
            vort_hat[i].name = "vort_hat_" + str(i)

        hel_hat = velocity_field_hat.cross_product(vort_hat)
        for i in jnp.arange(3):
            hel_hat[i].name = "hel_hat_" + str(i)
        return (vort_hat, hel_hat)

    def get_flow_rate(self):
        vel_hat = self.get_latest_field("velocity_hat")
        return vel_hat[0].no_hat().definite_integral(1)[0, 0]
        # return vel_hat[0].no_hat().volume_integral()/(self.domain.scale_factors[0] * self.domain.scale_factors[2])

    def update_flow_rate(self):
        self.flow_rate = self.get_flow_rate()
        # dPdx = -self.flow_rate * 3 / 2 / (self.Re_tau * self.u_max_over_u_tau)
        dPdx = -self.flow_rate * 3 / 2 / self.Re_tau
        # dPdx = -self.flow_rate / self.Re
        self.dpdx = Field.FromFunc(
            self.domain_no_hat, lambda X: dPdx + 0.0 * X[0] * X[1] * X[2]
        ).hat()
        self.dpdz = Field.FromFunc(
            self.domain_no_hat, lambda X: 0.0 + 0.0 * X[0] * X[1] * X[2]
        ).hat()

    def update_nonlinear_terms(self, velocity_field=None):
        if type(velocity_field) == NoneType:
            velocity_field_ = self.get_latest_field("velocity_hat")
        else:
            velocity_field_ = velocity_field
        (
            h_v_hat,
            h_g_hat,
            vort_hat,
            conv_ns_hat,
        ) = self.nonlinear_update_fn(
            self.domain_no_hat,
            jnp.array(
                [
                    velocity_field_[0].field,
                    velocity_field_[1].field,
                    velocity_field_[2].field,
                ]
            ),
        )
        h_v_hat_field = Field(self.domain_no_hat, h_v_hat, name="h_v_hat")
        h_g_hat_field = Field(self.domain_no_hat, h_g_hat, name="h_g_hat")
        vort_hat_field = Field(self.domain_no_hat, vort_hat, name="vort_hat")
        conv_ns_hat_field = Field(self.domain_no_hat, conv_ns_hat, name="conv_ns_hat")
        try:
            self.append_field("h_v_hat", h_v_hat_field)
            self.append_field("h_g_hat", h_g_hat_field)
            self.append_field("vort_hat", vort_hat_field)
            self.append_field("conv_ns_hat", conv_ns_hat_field)
            return (h_v_hat, h_g_hat, vort_hat, conv_ns_hat)
        except AttributeError:
            return (h_v_hat_field, h_g_hat_field, vort_hat_field, conv_ns_hat_field)

    def get_cheb_mat_2_homogeneous_dirichlet(self):
        return self.get_initial_field("velocity_hat")[
            0
        ].get_cheb_mat_2_homogeneous_dirichlet(1)

    def get_cheb_mat_2_homogeneous_dirichlet_only_rows(self):
        return self.domain_no_hat.get_cheb_mat_2_homogeneous_dirichlet_only_rows(1)

    def get_time_step(self):
        dX = self.domain_no_hat.grid[0][1:] - self.domain_no_hat.grid[0][:-1]
        dY = self.domain_no_hat.grid[1][1:] - self.domain_no_hat.grid[1][:-1]
        dZ = self.domain_no_hat.grid[2][1:] - self.domain_no_hat.grid[2][:-1]
        DX, DY, DZ = jnp.meshgrid(dX, dY, dZ, indexing="ij")
        vel = self.get_latest_field("velocity_hat").no_hat()
        U = vel[0][1:, 1:, 1:]
        V = vel[1][1:, 1:, 1:]
        W = vel[2][1:, 1:, 1:]
        u_cfl = (abs(DX) / abs(U)).min().real
        v_cfl = (abs(DY) / abs(V)).min().real
        w_cfl = (abs(DZ) / abs(W)).min().real
        return min(self.max_dt, self.max_cfl * min([u_cfl, v_cfl, w_cfl]))

    def get_rk_parameters(self):
        return (
            [29 / 96, -3 / 40, 1 / 6],
            [37 / 160, 5 / 24, 1 / 6],
            [8 / 15, 5 / 12, 3 / 4],
            [0, -17 / 60, -5 / 12],
        )

    def assemble_rk_matrices(self, Ly, kx, kz, i):
        alpha, beta, _, _ = self.get_rk_parameters()
        n = Ly.shape[0]
        I = jnp.eye(n)
        L = Ly + I * (-(kx**2 + kz**2)) / self.Re_tau
        lhs_mat = I - beta[i] * self.dt * L
        rhs_mat = I + alpha[i] * self.dt * L
        return (lhs_mat, rhs_mat)

    def prepare(self):
        self.poisson_mat = self.get_initial_field("velocity_hat")[
            0
        ].assemble_poisson_matrix()

    def perform_runge_kutta_step(self):
        self.dt = self.get_time_step()
        Re = self.Re_tau
        vel_hat = self.get_latest_field("velocity_hat")

        # start runge-kutta stepping
        _, _, gamma, xi = self.get_rk_parameters()

        D2 = jnp.linalg.matrix_power(self.domain_no_hat.diff_mats[1], 2)
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        # n = D2_hom_diri.shape[0]
        n = D2.shape[0]
        I = jnp.eye(n)
        Z = jnp.zeros((n, n))

        L_NS = 1 / Re * jnp.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])

        def perform_single_rk_step_for_single_wavenumber(
            step,
            v_1_lap_hat,
            vort_hat,
            conv_ns_hat,
            conv_ns_hat_old,
            h_v_hat,
            h_g_hat,
            h_v_hat_old,
            h_g_hat_old,
        ):
            def fn(K):
                domain = self.domain
                kx = K[0]
                kz = K[1]
                kx_ = domain.grid[0][kx]
                kz_ = domain.grid[2][kz]

                # wall-normal velocity
                # p-part
                L = 1 / Re * D2
                lhs_mat_p, rhs_mat_p = self.assemble_rk_matrices(L, kx_, kz_, step)

                phi_hat_lap = v_1_lap_hat[kx, :, kz]

                N_new = h_v_hat[kx, :, kz]
                if type(h_v_hat_old == NoneType):
                    N_old = N_new
                else:
                    N_old = h_v_hat_old[kx, :, kz]
                rhs_p = rhs_mat_p @ phi_hat_lap + (self.dt * gamma[step]) * N_new + (self.dt * xi[step]) * N_old
                lhs_mat_p = domain.enforce_homogeneous_dirichlet(lhs_mat_p)
                rhs_p = domain.update_boundary_conditions_fourier_field_slice(
                    rhs_p, 1
                )

                phi_hat_lap_new = jnp.linalg.inv(lhs_mat_p) @ rhs_p


                v_1_lap_hat_new_p = phi_hat_lap_new

                # compute velocity in y direction
                v_1_lap_hat_new_p = (
                    domain.update_boundary_conditions_fourier_field_slice(
                        v_1_lap_hat_new_p, 1
                    )
                )
                v_1_hat_new_p = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_p, self.poisson_mat, kx, kz
                )
                v_1_hat_new_p = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_p, 1
                )

                # a-part - numerical solution
                L_a_y = 1 / Re * D2

                L_a = L_a_y + I * (-(kx_**2 + kz_**2)) / Re

                lhs_mat_a, _ = self.assemble_rk_matrices(L_a, kx_, kz_, step)
                rhs_a = jnp.zeros(n)
                lhs_mat_a, rhs_a = domain.enforce_inhomogeneous_dirichlet(lhs_mat_a, rhs_a, 0.0, 1.0)
                phi_a_hat_new = jnp.linalg.inv(lhs_mat_a) @ rhs_a
                v_1_lap_hat_new_a = phi_a_hat_new

                # compute velocity in y direction
                v_1_hat_new_a = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_a, self.poisson_mat, kx, kz
                )
                v_1_hat_new_a = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_a, 1
                )

                v_1_hat_new_b = jnp.flip(v_1_hat_new_a)

                # reconstruct velocity s.t. hom. Neumann is fulfilled
                v_1_hat_new_p_diff = domain.diff_fourier_field_slice(v_1_hat_new_p, 1)
                v_1_hat_new_a_diff = domain.diff_fourier_field_slice(v_1_hat_new_a, 1)
                v_1_hat_new_b_diff = domain.diff_fourier_field_slice(v_1_hat_new_b, 1)
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
                L = 1 / Re * D2
                lhs_mat_vort, rhs_mat_vort = self.assemble_rk_matrices(L, kx_, kz_, step)



                vort_1_hat = vort_hat[1]
                phi_vort_hat = vort_1_hat[kx, :, kz]

                N_new = h_g_hat[kx, :, kz]
                if type(h_g_hat_old == NoneType):
                    N_old = N_new
                else:
                    N_old = h_g_hat_old[kx, :, kz]

                rhs_vort = rhs_mat_vort @ phi_vort_hat + (self.dt * gamma[step]) * N_new + (self.dt * xi[step]) * N_old

                lhs_mat_vort = domain.enforce_homogeneous_dirichlet(lhs_mat_vort)
                rhs_vort = domain.update_boundary_conditions_fourier_field_slice(
                    rhs_vort, 1
                )

                phi_hat_vort_new = jnp.linalg.inv(lhs_mat_vort) @ (
                    rhs_vort
                )

                vort_1_hat_new = phi_hat_vort_new

                # compute velocities in x and z directions
                def rk_00():
                    kx__ = 0
                    kz__ = 0
                    lhs_mat_00, rhs_mat_00 = self.assemble_rk_matrices(
                        L_NS, 0, 0, step
                    )
                    v_hat = jnp.block(
                        [vel_hat[0][kx__, :, kz__], vel_hat[2][kx__, :, kz__]]
                    )
                    N_00_new = jnp.block(
                        [
                            -conv_ns_hat[0][kx__, :, kz__],
                            -conv_ns_hat[2][kx__, :, kz__],
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
                        N_00_old = jnp.block(
                            [
                                -conv_ns_hat_old[0][kx__, :, kz__],
                                -conv_ns_hat_old[2][kx__, :, kz__],
                            ]
                        ) + jnp.block(
                            [
                                -self.dpdx[kx__, :, kz__],
                                -self.dpdz[kx__, :, kz__],
                            ]
                        )
                    v_hat_new = jnp.linalg.inv(lhs_mat_00) @ (
                        rhs_mat_00 @ v_hat
                        + (self.dt * gamma[step]) * N_00_new
                        + (self.dt * xi[step]) * N_00_old
                    )
                    return (v_hat_new[:n], v_hat_new[n:])

                def rk_not_00(kx, kz):
                    kx_ = domain.grid[0][kx]
                    kz_ = domain.grid[2][kz]
                    minus_kx_kz_sq = -(kx_**2 + kz_**2)
                    v_1_new_y = domain.diff_fourier_field_slice(v_1_hat_new, 1)
                    v_0_new = (
                        -1j * kx_ * v_1_new_y + 1j * kz_ * vort_1_hat_new
                    ) / minus_kx_kz_sq
                    v_2_new = (
                        -1j * kz_ * v_1_new_y - 1j * kx_ * vort_1_hat_new
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
                return (v_0_new_field, v_1_hat_new, v_2_new_field, v_1_lap_hat_new_a)

            return fn

        number_of_rk_steps = 3

        vel_new_hat = vel_hat

        h_v_hat_old, h_g_hat_old, conv_ns_hat_old = (None, None, None)

        for step in range(number_of_rk_steps):
            # update nonlinear terms
            (
                h_v_hat,
                h_g_hat,
                vort_hat,
                conv_ns_hat,
            ) = self.nonlinear_update_fn(
                self.domain_no_hat,
                jnp.array([vel_hat[0].field, vel_hat[1].field, vel_hat[2].field]),
            )

            # solve equations
            v_1_hat = vel_hat[1]
            v_1_lap_hat = v_1_hat.laplacian()
            vel_new_hat, _ = vel_hat.reconstruct_from_wavenumbers(
                perform_single_rk_step_for_single_wavenumber(
                    step,
                    v_1_lap_hat,
                    vort_hat,
                    conv_ns_hat,
                    conv_ns_hat_old,
                    h_v_hat,
                    h_g_hat,
                    h_v_hat_old,
                    h_g_hat_old,
                ),
            )
            h_v_hat_old, h_g_hat_old, conv_ns_hat_old = (
                h_v_hat,
                h_g_hat,
                conv_ns_hat,
            )

            vel_new_hat.update_boundary_conditions()
            vel_hat = vel_new_hat

        vel_new_hat.name = "velocity_hat"
        for i in jnp.arange(len(vel_new_hat)):
            vel_new_hat[i].name = "velocity_hat_" + ["x", "y", "z"][i]
        self.append_field("velocity_hat", vel_new_hat)

    def perform_cn_ab_step(self):
        self.dt = self.get_time_step()
        dt = self.dt

        Re = self.Re_tau
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        D2 = jnp.linalg.matrix_power(self.domain_no_hat.diff_mats[1], 2)
        n = D2.shape[0]
        Z = jnp.zeros((n, n))
        I = jnp.eye(n)
        L_NS_y = 1 / Re * jnp.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])
        I = jnp.eye(n)

        vel_hat = self.get_latest_field("velocity_hat")

        def perform_single_cn_ab_step_for_single_wavenumber(
            v_1_lap_hat_p,
            vort_hat,
            conv_ns_hat,
            conv_ns_hat_old,
            h_v_hat,
            h_g_hat,
            h_v_hat_old,
            h_g_hat_old,
        ):
            def fn(K):
                domain = self.domain
                kx = K[0]
                kz = K[1]
                kx_ = domain.grid[0][kx]
                kz_ = domain.grid[2][kz]

                # wall-normal velocity
                # p-part
                L_p_y = 1 / Re * D2

                L_p = L_p_y + I * (-(kx_**2 + kz_**2)) / Re

                phi_p_hat = v_1_lap_hat_p[kx, :, kz]

                N_p_new = h_v_hat[kx, :, kz]
                N_p_old = h_v_hat_old[kx, :, kz]
                rhs_mat_p = I + dt / 2 * L_p
                lhs_mat_p = I - dt / 2 * L_p
                lhs_mat_p = domain.enforce_homogeneous_dirichlet(lhs_mat_p)
                rhs = rhs_mat_p @ phi_p_hat + dt / 2 * (3 * N_p_new - N_p_old)
                rhs = domain.update_boundary_conditions_fourier_field_slice(
                    rhs, 1
                )
                phi_p_hat_new = jnp.linalg.inv(lhs_mat_p) @ rhs
                v_1_lap_hat_new_p = phi_p_hat_new

                # compute velocity in y direction
                v_1_lap_hat_new_p = (
                    domain.update_boundary_conditions_fourier_field_slice(
                        v_1_lap_hat_new_p, 1
                    )
                )
                v_1_hat_new_p = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_p, self.poisson_mat, kx, kz
                )
                v_1_hat_new_p = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_p, 1
                )

                # a-part (numerical solution)
                L_a_y = 1 / Re * D2

                L_a = L_a_y + I * (-(kx_**2 + kz_**2)) / Re

                lhs_mat_a = I - dt / 2 * L_a
                rhs_a = jnp.zeros(n)
                lhs_mat_a, rhs_a = domain.enforce_inhomogeneous_dirichlet(lhs_mat_a, rhs_a, 0.0, 1.0)
                phi_a_hat_new = jnp.linalg.inv(lhs_mat_a) @ rhs_a
                v_1_lap_hat_new_a = phi_a_hat_new

                # compute velocity in y direction
                v_1_hat_new_a = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_a, self.poisson_mat, kx, kz
                )
                v_1_hat_new_a = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_a, 1
                )
                v_1_hat_new_b = jnp.flip(v_1_hat_new_a)

                # reconstruct velocity s.t. hom. Neumann is fulfilled
                v_1_hat_new_p_diff = domain.diff_fourier_field_slice(v_1_hat_new_p, 1)
                v_1_hat_new_a_diff = domain.diff_fourier_field_slice(v_1_hat_new_a, 1)
                v_1_hat_new_b_diff = domain.diff_fourier_field_slice(v_1_hat_new_b, 1)
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
                L_vort_y = 1 / Re * D2
                L_vort = L_vort_y + I * (-(kx_**2 + kz_**2)) / Re

                rhs_mat_vort = I + dt / 2 * L_vort
                lhs_mat_vort = I - dt / 2 * L_vort
                lhs_mat_vort = domain.enforce_homogeneous_dirichlet(lhs_mat_vort)
                vort_1_hat = vort_hat[1]
                phi_vort_hat = vort_1_hat[kx, :, kz]

                N_vort_new = h_g_hat[kx, :, kz]
                N_vort_old = h_g_hat_old[kx, :, kz]
                rhs_vort = rhs_mat_vort @ phi_vort_hat + dt / 2 * (3 * N_vort_new - N_vort_old)
                rhs_vort = domain.update_boundary_conditions_fourier_field_slice(
                    rhs_vort, 1
                )
                phi_vort_hat_new = jnp.linalg.solve(
                    lhs_mat_vort,
                    rhs_vort
                )
                vort_1_hat_new = phi_vort_hat_new

                # compute velocities in x and z directions
                def rk_00():
                    kx__ = 0
                    kz__ = 0

                    v_hat = jnp.block(
                        [vel_hat[0][kx__, :, kz__], vel_hat[2][kx__, :, kz__]]
                    )
                    I_ = jnp.eye(2 * n)
                    L_NS = L_NS_y + I_ * (-(kx_**2) - kz_**2) / Re
                    rhs_mat_ns = I_ + dt / 2 * L_NS
                    lhs_mat_ns = I_ - dt / 2 * L_NS
                    N_00_new = (
                        jnp.block(
                            [
                                -conv_ns_hat[0][kx__, :, kz__],
                                -conv_ns_hat[2][kx__, :, kz__],
                            ]
                        )
                        + jnp.block(
                            [
                                -self.dpdx[kx__, :, kz__],
                                -self.dpdz[kx__, :, kz__],
                            ]
                        )
                    )
                    N_00_old = (
                        jnp.block(
                            [
                                -conv_ns_hat_old[0][kx__, :, kz__],
                                -conv_ns_hat_old[2][kx__, :, kz__],
                            ]
                        )
                        + jnp.block(
                            [
                                -self.dpdx[kx__, :, kz__],
                                -self.dpdz[kx__, :, kz__],
                            ]
                        )
                    )
                    rhs_ns = rhs_mat_ns @ v_hat + dt / 2 * (3 * N_00_new - N_00_old)
                    rhs_ns = jnp.block([0.0, rhs_ns[1:n-1], 0.0 ,0.0, rhs_ns[n+1:-1], 0.0])
                    v_hat_new = jnp.linalg.solve(
                        lhs_mat_ns,
                        rhs_ns
                    )
                    return (v_hat_new[:n], v_hat_new[n:])

                def rk_not_00(kx, kz):
                    kx_ = domain.grid[0][kx]
                    kz_ = domain.grid[2][kz]
                    minus_kx_kz_sq = -(kx_**2 + kz_**2)
                    v_1_new_y = domain.diff_fourier_field_slice(v_1_hat_new, 1)
                    v_0_new = (
                        -1j * kx_ * v_1_new_y + 1j * kz_ * vort_1_hat_new
                    ) / minus_kx_kz_sq
                    v_2_new = (
                        -1j * kz_ * v_1_new_y - 1j * kx_ * vort_1_hat_new
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
                return (v_0_new_field, v_1_hat_new, v_2_new_field, v_1_lap_hat_new_p)

            return fn

        v_1_lap_hat = vel_hat[1].laplacian()

        h_v_hat_0 = self.get_latest_field("h_v_hat").field
        h_g_hat_0 = self.get_latest_field("h_g_hat").field
        vort_hat_0 = self.get_latest_field("vort_hat").field
        conv_ns_hat_0 = self.get_latest_field("conv_ns_hat").field

        h_v_hat_0_old = self.get_field("h_v_hat", max(0, self.time_step - 1)).field
        h_g_hat_0_old = self.get_field("h_g_hat", max(0, self.time_step - 1)).field
        conv_ns_hat_0_old = self.get_field("conv_ns_hat", max(0, self.time_step - 1)).field

        # solve equations
        vel_new_hat, _ = vel_hat.reconstruct_from_wavenumbers(
            perform_single_cn_ab_step_for_single_wavenumber(
                # v_1_lap_hat_p,
                v_1_lap_hat,
                vort_hat_0,
                conv_ns_hat_0,
                conv_ns_hat_0_old,
                h_v_hat_0,
                h_g_hat_0,
                h_v_hat_0_old,
                h_g_hat_0_old,
            ),
        )
        vel_new_hat.update_boundary_conditions()

        vel_new_hat.name = "velocity_hat"
        for i in jnp.arange(len(vel_new_hat)):
            vel_new_hat[i].name = "velocity_hat_" + ["x", "y", "z"][i]
        self.append_field("velocity_hat", vel_new_hat)

        self.update_nonlinear_terms()

    def perform_time_step(self):
        # return self.perform_runge_kutta_step() # TODO not working yet
        return self.perform_cn_ab_step()


def solve_navier_stokes_laminar(
    Re=1.8e2,
    end_time=1e1,
    max_iter=10000,
    Nx=6,
    Ny=40,
    Nz=None,
    pertubation_factor=0.1,
    scale_factors=(1.87, 1.0, 0.93),
):
    Ny = Ny
    Nz = Nz or Nx + 4

    domain = Domain((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)
    # domain = Domain((Nx, Ny, Nz), (True, False, True))

    vel_x_fn_ana = (
        lambda X: -1 * NavierStokesVelVort.u_max_over_u_tau * (X[1] + 1) * (X[1] - 1)
        + 0.0 * X[0] * X[2]
    )
    vel_x_ana = Field.FromFunc(domain, vel_x_fn_ana, name="vel_x_ana")

    vel_x_fn = lambda X: jnp.pi / 3 * NavierStokesVelVort.u_max_over_u_tau * (
        pertubation_factor
        * jnp.cos(X[1] * jnp.pi / 2)
        * (jnp.cos(3 * X[0]) ** 2 * jnp.cos(4 * X[2]) ** 2)
    ) + (1 - pertubation_factor) * vel_x_fn_ana(X)

    # add small pertubation in y and z to see if it decays
    vel_y_fn = (
        lambda X: 0.1
        * pertubation_factor
        * NavierStokesVelVort.u_max_over_u_tau
        * (
            jnp.pi
            / 3
            # * jnp.cos(X[1] * jnp.pi / 2)
            * (1 - X[1] ** 2) ** 2
            * jnp.cos(3 * X[0])
            * jnp.cos(4 * X[2])
        )
    )
    vel_z_fn = (
        lambda X: 0.1
        * jnp.pi
        / 3
        * pertubation_factor
        * NavierStokesVelVort.u_max_over_u_tau
        * (jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(5 * X[0]) * jnp.cos(3 * X[2]))
    )
    vel_x = Field.FromFunc(domain, vel_x_fn, name="vel_x")
    vel_y = Field.FromFunc(domain, vel_y_fn, name="vel_y")
    vel_z = Field.FromFunc(domain, vel_z_fn, name="vel_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")

    nse = NavierStokesVelVort.FromVelocityField(vel, Re)
    nse.end_time = end_time
    nse.max_iter = max_iter
    vel_0 = nse.get_initial_field("velocity_hat").no_hat()

    plot_interval = 10

    def before_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_field("velocity_hat", i).no_hat()
            vel_old = nse.get_field("velocity_hat", i - 1).no_hat()
            vel[0].plot_center(1, vel_0[0], vel_x_ana)
            vel[1].plot_center(1, vel_0[1])
            vel[2].plot_center(1, vel_0[2])
            vel[0].plot_3d()
            vel[1].plot_3d()
            vel[2].plot_3d()
            print(abs(vel[0] - vel_x_ana))
            print(abs(vel[1]))
            print(abs(vel[2]))
            old_error = abs(vel_old[0] - vel_x_ana)
            new_error = abs(vel[0] - vel_x_ana)
            rel_change = abs(new_error - old_error) / old_error
            print("rel_change: " + str(rel_change))

    nse.before_time_step_fn = before_time_step

    return nse

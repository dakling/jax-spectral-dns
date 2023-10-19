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
    pass
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    pass
from field import Field, VectorField, FourierField, FourierFieldSlice

try:
    reload(sys.modules["equation"])
except:
    pass
from equation import Equation


# @partial(jax.jit, static_argnums=0)
def update_nonlinear_terms_high_performance(domain, vel_hat_new, vel_base_hat):
    vel_new = jnp.array(
        [
            domain.no_hat(vel_hat_new.at[i].get())
            for i in jnp.arange(domain.number_of_dimensions)
        ]
    )
    vort_new = domain.curl(vel_new)

    vel_new_sq = 0
    for i in domain.all_dimensions():
        vel_new_sq += vel_new[i] * vel_new[i]
    vel_new_sq_nabla = []
    for i in domain.all_dimensions():
        vel_new_sq_nabla.append(domain.diff(vel_new_sq, i))

    hel_new_ = domain.cross_product(vel_new, vort_new) + 1 / 2 * jnp.array(vel_new_sq_nabla)

    # a-term
    vel_base = jnp.array(
        [
            domain.no_hat(vel_base_hat.at[i].get())
            for i in jnp.arange(domain.number_of_dimensions)
        ]
    )
    vel_new_sq_a = 0
    for i in domain.all_dimensions():
        vel_new_sq_a += vel_new[i] * vel_base[i]
    vel_new_sq_nabla_a = []
    for i in domain.all_dimensions():
        vel_new_sq_nabla_a.append(domain.diff(vel_new_sq_a, i))
    hel_new_a = domain.cross_product(vel_base, vort_new) + 1 / 2 * jnp.array(vel_new_sq_nabla_a)

    # b-term
    vort_b = domain.curl(vel_base)
    vel_new_sq_nabla_b = vel_new_sq_nabla_a
    hel_new_b = domain.cross_product(vel_new, vort_b) + 1 / 2 * jnp.array(vel_new_sq_nabla_b)

    # exact expression
    hel_new = hel_new_ + hel_new_a + hel_new_b
    # linearized expression
    # hel_new = hel_new_a + hel_new_b

    h_v_new = (
        - domain.diff(domain.diff(hel_new[0], 0) + domain.diff(hel_new[2], 2), 1)
        + domain.diff(hel_new[1], 0, 2)
        + domain.diff(hel_new[1], 2, 2)
    )
    h_g_new = (
        domain.diff(hel_new[0], 2) - domain.diff(hel_new[2], 0)
    )

    # conv_ns_new = []
    # for i in domain.all_dimensions():
    #     out = 0
    #     for k in domain.all_dimensions():
    #         # out += vel_new[k] * domain.diff(vel_new[i], k)
    #         out += vel_base[k] * domain.diff(vel_new[i], k)
    #         out += vel_new[k] * domain.diff(vel_base[i], k)
    #     conv_ns_new.append(out)

    h_v_hat_new = domain.field_hat(h_v_new)
    h_g_hat_new = domain.field_hat(h_g_new)
    vort_hat_new = [domain.field_hat(vort_new[i]) for i in domain.all_dimensions()]
    conv_ns_hat_new = [domain.field_hat(hel_new[i]) for i in domain.all_dimensions()]

    return (h_v_hat_new, h_g_hat_new, vort_hat_new, conv_ns_hat_new)


class NavierStokesVelVortPertubation(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation) for pertubations on top of a base flow."
    # max_cfl = 0.7
    max_cfl = 0.7 / 10
    # max_dt = 1e10
    max_dt = 1e-2

    def __init__(self, shape, velocity_field, **params):
        domain = velocity_field[0].domain
        self.domain_no_hat = velocity_field[0].domain_no_hat

        velocity_x_base = Field.FromFunc(self.domain_no_hat, lambda X: (1 - X[1]**2) + 0.0 * X[0] * X[2], name="velocity_x_base")
        velocity_y_base = Field.FromFunc(self.domain_no_hat, lambda X: 0.0 * X[0] * X[1] * X[2], name="velocity_y_base")
        velocity_z_base = Field.FromFunc(self.domain_no_hat, lambda X: 0.0 * X[0] * X[1] * X[2], name="velocity_z_base")
        velocity_base_hat = VectorField([velocity_x_base.hat(), velocity_y_base.hat(), velocity_z_base.hat()])
        velocity_base_hat.name = "velocity_base_hat"
        # v_1_lap_hat_a = FourierField.FromFunc(self.domain_no_hat, lambda X: (X[1] + 1)/2 + 0.0 * X[0] * X[2])
        # v_1_lap_hat_a.name="v_1_lap_hat_a"
        # super().__init__(domain, velocity_field, velocity_base_hat, v_1_lap_hat_a, **params)
        super().__init__(domain, velocity_field, velocity_base_hat, **params)
        self.Re = params["Re"]
        self.flow_rate = self.get_flow_rate()
        self.dt = self.get_time_step()
        self.poisson_mat = None
        self.lhs_mat_inv = []
        self.rhs_mat = []

    @classmethod
    def FromVelocityField(cls, shape, velocity_field, Re=1.8e2, end_time=1e0):
        velocity_field_hat = velocity_field.hat()
        velocity_field_hat.name = "velocity_hat"
        return cls(shape, velocity_field_hat, Re=Re, end_time=end_time)

    @classmethod
    def FromRandom(cls, shape, Re, end_time=1e0):
        domain = Domain(shape, (True, False, True))
        vel_x = Field.FromRandom(domain, name="u0")
        vel_y = Field.FromRandom(domain, name="u1")
        vel_z = Field.FromRandom(domain, name="u2")
        # vel_y.update_boundary_conditions()
        vel = VectorField([vel_x, vel_y, vel_z], "velocity")
        return cls.FromVelocityField(shape, vel, Re, end_time=end_time)

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
        return vel_hat[0].no_hat().integrate(1, 1, 0.0).field[0, 0, 0]

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
        L = Ly + I * (-(kx**2 + kz**2))
        lhs_mat_inv = jnp.linalg.inv(I - beta[i] * self.dt * L)
        rhs_mat = I + alpha[i] * self.dt * L
        return (lhs_mat_inv, rhs_mat)

    def prepare(self):
        self.poisson_mat = self.get_initial_field("velocity_hat")[
            0
        ].assemble_poisson_matrix()

    def perform_runge_kutta_step(self):
        start_time = time.time()

        self.dt = self.get_time_step()
        Re = self.Re
        vel_hat = self.get_latest_field("velocity_hat")

        # start runge-kutta stepping
        alpha, beta, gamma, xi = self.get_rk_parameters()

        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        D2_hom_diri_only_rows = self.get_cheb_mat_2_homogeneous_dirichlet_only_rows()
        n = D2_hom_diri.shape[0]
        Z = jnp.zeros((n, n))

        dPdx = 0
        dPdz = 0
        L_NS = 1 / Re * jnp.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])

        def perform_single_rk_step_for_single_wavenumber(
            step,
            v_1_lap_hat,
            v_1_lap_hat_a,
            vort_hat,
            ns_conv_hat,
            ns_conv_hat_old,
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
                L = 1 / Re * D2_hom_diri
                lhs_mat_inv, rhs_mat = self.assemble_rk_matrices(L, kx_, kz_, step)

                phi_hat = v_1_lap_hat[kx, :, kz]

                N_new = h_v_hat[kx, :, kz]
                N_old = h_v_hat_old[kx, :, kz]
                phi_hat_new = lhs_mat_inv @ (
                    rhs_mat @ phi_hat
                    + (self.dt * gamma[step]) * N_new
                    + (self.dt * xi[step]) * N_old
                )

                v_1_lap_hat_new_p = phi_hat_new


                # compute velocity in y direction
                v_1_hat_new_p = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_hat_new_p, self.poisson_mat, kx, kz
                )
                v_1_hat_new_p = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_p, 1
                )

                # a-part
                L_a = 1 / Re * D2_hom_diri_only_rows

                # runge kutta
                def set_nth_mat_row_to_unit(matr, n):
                    N = matr.shape[0]
                    return jnp.block(
                        [[matr[:n, :]], [jnp.eye(N)[n,:]], [matr[n+1:, :]]]
                    )
                I = jnp.eye(L_a.shape[0])
                L_ = I - beta[step] * self.dt * L_a
                R_ = I + alpha[step] * self.dt * L_a
                for i in [0, n-1]:
                    L_ = set_nth_mat_row_to_unit(L_, i)
                    R_ = set_nth_mat_row_to_unit(R_, i)

                L_inv = jnp.linalg.inv(L_)

                phi_hat_a = jnp.block([1.0, v_1_lap_hat_a[kx, 1:-1, kz], 0.0])

                phi_hat_new_a = L_inv @ (
                    R_ @ phi_hat_a
                )

                # fig, ax = plt.subplots(1,1)
                # ax.plot(self.domain_no_hat.grid[1], phi_hat_a)
                # fig.savefig("./plots/" + "plot_v_1_lap_a_" + str(kx) + "_" + str(kz))
                v_1_lap_new_a = phi_hat_new_a
                # compute velocity in y direction
                v_1_hat_new_a = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_new_a, self.poisson_mat, kx, kz
                )
                v_1_hat_new_a = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new_a, 1
                )

                v_1_hat_new_b = jnp.flip(v_1_hat_new_a)

                # reconstruct velocity s.t. hom. Neumann is fulfilled
                v_1_hat_new_p_diff = domain.diff_fourier_field_slice(v_1_hat_new_p, 1)
                v_1_hat_new_a_diff = domain.diff_fourier_field_slice(v_1_hat_new_a, 1)
                v_1_hat_new_b_diff = domain.diff_fourier_field_slice(v_1_hat_new_b, 1)
                M = jnp.array([[v_1_hat_new_a_diff[0], v_1_hat_new_b_diff[0]],
                               [v_1_hat_new_a_diff[-1], v_1_hat_new_b_diff[-1]]])
                R = jnp.array([-v_1_hat_new_p_diff[0], -v_1_hat_new_p_diff[-1]])
                AB = jnp.linalg.solve(M, R)
                a, b = AB[0], AB[1]
                v_1_hat_new = v_1_hat_new_p + a * v_1_hat_new_a + b * v_1_hat_new_b
                v_1_hat_new = domain.update_boundary_conditions_fourier_field_slice(v_1_hat_new, 1)
                # if kx==1 and kz==0:
                #     fig, ax = plt.subplots(1,1)
                #     ys = domain.grid[1]
                #     ax.plot(ys, vel_hat[1][kx, :, kz])
                #     ax.plot(ys, v_1_hat_new)
                #     fig.savefig("plots/plot.pdf")
                #     raise Exception("break")

                # vorticity
                L = 1 / Re * D2_hom_diri
                lhs_mat_inv, rhs_mat = self.assemble_rk_matrices(L, kx_, kz_, step)

                vort_1_hat = vort_hat[1]
                phi_hat = vort_1_hat[kx, :, kz]

                N_new = h_g_hat[kx, :, kz]
                N_old = h_g_hat_old[kx, :, kz]
                phi_hat_new = lhs_mat_inv @ (
                    rhs_mat @ phi_hat
                    + (self.dt * gamma[step]) * N_new
                    + (self.dt * xi[step]) * N_old
                )

                vort_1_hat_new = phi_hat_new

                # compute velocities in x and z directions
                def rk_00():
                    kx__ = 0
                    kz__ = 0
                    lhs_mat_00_inv, rhs_mat_00 = self.assemble_rk_matrices(
                        L_NS, 0, 0, step
                    )
                    v_hat = jnp.block(
                        [vel_hat[0][kx__, :, kz__], vel_hat[2][kx__, :, kz__]]
                    )
                    Nx = len(domain.grid[0])
                    Nz = len(domain.grid[2])
                    dx = Nx * (2 * jnp.pi / domain.scale_factors[0]) ** (2)
                    dz = Nz * (2 * jnp.pi / domain.scale_factors[2]) ** (2)
                    N_00_new = (
                        jnp.block(
                            [-ns_conv_hat[0][kx__, :, kz__], -ns_conv_hat[2][kx__, :, kz__]]
                        )
                    )
                    N_00_old = (
                        jnp.block(
                            [
                                -ns_conv_hat_old[0][kx__, :, kz__],
                                -ns_conv_hat_old[2][kx__, :, kz__],
                            ]
                        )
                    )
                    v_hat_new = lhs_mat_00_inv @ (
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
                return (v_0_new_field, v_1_hat_new, v_2_new_field, v_1_lap_new_a)

            return fn


        vel_base_hat = self.get_latest_field("velocity_base_hat")
        # perform first RK step
        v_1_hat_0 = vel_hat[1]
        v_1_lap_hat_0 = v_1_hat_0.laplacian()
        v_1_lap_hat_a_0 = self.get_latest_field("v_1_lap_hat_a")

        # jit_update = jax.jit(self.update_nonlinear_terms_high_performance)
        # h_v_hat_0, h_g_hat_0, vort_hat_0, hel_hat_0 = self.update_nonlinear_terms(
        #     vel_hat
        # )
        (
            h_v_hat_0,
            h_g_hat_0,
            vort_hat_0,
            ns_conv_hat_0,
        ) = update_nonlinear_terms_high_performance(
            self.domain_no_hat,
            jnp.array([vel_hat[0].field, vel_hat[1].field, vel_hat[2].field]),
            jnp.array(
                [vel_base_hat[0].field, vel_base_hat[1].field, vel_base_hat[2].field]
            ),
        )

        # solve equations
        vel_new_hat_1, other_field_1 = vel_hat.reconstruct_from_wavenumbers(
            perform_single_rk_step_for_single_wavenumber(
                0,
                v_1_lap_hat_0,
                v_1_lap_hat_a_0,
                vort_hat_0,
                ns_conv_hat_0,
                ns_conv_hat_0,
                h_v_hat_0,
                h_g_hat_0,
                h_v_hat_0,
                h_g_hat_0,
            ),
            1
        )

        vel_new_hat_1.update_boundary_conditions()
        v_1_lap_a_new_1 = other_field_1[0]

        # update nonlinear terms
        (
            h_v_hat_1,
            h_g_hat_1,
            vort_hat_1,
            ns_conv_hat_1,
        ) = update_nonlinear_terms_high_performance(
            self.domain_no_hat,
            jnp.array(
                [vel_new_hat_1[0].field, vel_new_hat_1[1].field, vel_new_hat_1[2].field]
            ),
            jnp.array(
                [vel_base_hat[0].field, vel_base_hat[1].field, vel_base_hat[2].field]
            ),
        )

        # perform second RK step
        v_1_hat_1 = vel_new_hat_1[1]
        v_1_lap_hat_1 = v_1_hat_1.laplacian()

        # solve equations
        vel_new_hat_2, other_field_2 = vel_hat.reconstruct_from_wavenumbers(
            perform_single_rk_step_for_single_wavenumber(
                1,
                v_1_lap_hat_1,
                v_1_lap_a_new_1,
                vort_hat_1,
                ns_conv_hat_1,
                ns_conv_hat_0,
                h_v_hat_1,
                h_g_hat_1,
                h_v_hat_0,
                h_g_hat_0,
            ),
            1
        )
        vel_new_hat_2.update_boundary_conditions()
        v_1_lap_a_new_2 = other_field_2[0]
        # update nonlinear terms
        (
            h_v_hat_2,
            h_g_hat_2,
            vort_hat_2,
            ns_conv_hat_2,
        ) = update_nonlinear_terms_high_performance(
            self.domain_no_hat,
            jnp.array(
                [vel_new_hat_2[0].field, vel_new_hat_2[1].field, vel_new_hat_2[2].field]
            ),
            jnp.array(
                [vel_base_hat[0].field, vel_base_hat[1].field, vel_base_hat[2].field]
            ),
        )

        # perform third RK step
        v_1_hat_2 = vel_new_hat_2[1]
        v_1_lap_hat_2 = v_1_hat_2.laplacian()

        # solve equations
        vel_new_hat, other_field  = vel_hat.reconstruct_from_wavenumbers(
            perform_single_rk_step_for_single_wavenumber(
                2,
                v_1_lap_hat_2,
                v_1_lap_a_new_2,
                vort_hat_2,
                ns_conv_hat_2,
                ns_conv_hat_1,
                h_v_hat_2,
                h_g_hat_2,
                h_v_hat_1,
                h_g_hat_1,
            ),
            1
        )
        vel_new_hat.update_boundary_conditions()
        v_1_lap_a_new = other_field[0]

        vel_new_hat.name = "velocity_hat"
        for i in jnp.arange(len(vel_new_hat)):
            vel_new_hat[i].name = "velocity_hat_" + ["x", "y", "z"][i]
        self.append_field("velocity_hat", vel_new_hat)
        v_1_lap_a_new.name = "v_1_lap_hat_a"
        self.append_field("v_1_lap_hat_a", v_1_lap_a_new)

    def perform_cn_ab_step(self):
        self.dt = self.get_time_step()
        dt = self.dt

        Re = self.Re
        dPdx = 0
        dPdz = 0
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet_only_rows()
        D2_hom_diri_with_cols = self.get_cheb_mat_2_homogeneous_dirichlet()
        # D2_hom_diri_only_rows = self.get_cheb_mat_2_homogeneous_dirichlet_only_rows()
        n = D2_hom_diri.shape[0]
        Z = jnp.zeros((n, n))
        I = jnp.eye(n)
        L_NS = 1 / Re * jnp.block([[D2_hom_diri_with_cols, Z], [Z, D2_hom_diri_with_cols]])
        I = jnp.eye(n)

        vel_hat = self.get_latest_field("velocity_hat")
        vel_base_hat = self.get_latest_field("velocity_base_hat")
        vel_hat_old = self.get_field("velocity_hat", max(0, self.time_step - 1))

        def perform_single_cn_ab_step_for_single_wavenumber(
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
                # L_p_y = 1 / Re * D2_hom_diri
                L_p_y = 1 / Re * D2_hom_diri_with_cols

                L_p = L_p_y + I * (-(kx_**2 + kz_**2))

                phi_p_hat = v_1_lap_hat[kx, :, kz]

                N_p_new = -h_v_hat[kx, :, kz]
                N_p_old = -h_v_hat_old[kx, :, kz]
                rhs_mat_p = I + dt / (2 * Re) * L_p
                lhs_mat_p = I - dt / (2 * Re) * L_p
                phi_p_hat_new = jnp.linalg.solve(
                    lhs_mat_p, rhs_mat_p @ phi_p_hat + dt / 2 * (3 * N_p_new - N_p_old)
                )
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

                # a-part -> can be solved analytically
                denom = jnp.sqrt((1 - dt/2*Re * (- kx_**2 - kz_**2)))
                k = jnp.sqrt(2 * Re / dt) / denom
                A = jnp.exp(2*k) / (jnp.exp(3*k) - jnp.exp(k))
                B = - 1 / (jnp.exp(3*k) - jnp.exp(k))
                fn_ana = lambda y: jax.lax.cond(
                    abs(denom) > 1e-30,
                    lambda _: (A * jnp.exp(k * y) + B * jnp.exp(- k * y)),
                    lambda _: ((y + 1)/2),
                    denom)

                ys = self.domain_no_hat.grid[1]
                # phi_a_hat_new = jnp.array(list(map(fn_ana, ys)))
                phi_a_hat_new = jax.lax.map(fn_ana, ys)

                v_1_lap_new_a = phi_a_hat_new
                # compute velocity in y direction
                v_1_hat_new_a = domain.solve_poisson_fourier_field_slice(
                    v_1_lap_new_a, self.poisson_mat, kx, kz
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
                AB = jnp.linalg.solve(M, R)
                a, b = AB[0], AB[1]
                v_1_hat_new = v_1_hat_new_p + a * v_1_hat_new_a + b * v_1_hat_new_b
                v_1_hat_new = domain.update_boundary_conditions_fourier_field_slice(
                    v_1_hat_new, 1
                )

                # vorticity
                L_vort_y = 1 / Re * D2_hom_diri_with_cols
                L_vort = L_vort_y + I * (-(kx_**2 + kz_**2))

                rhs_mat_vort = I + dt / (2 * Re) * L_vort
                lhs_mat_vort = I - dt / (2 * Re) * L_vort
                vort_1_hat = vort_hat[1]
                phi_vort_hat = vort_1_hat[kx, :, kz]

                N_vort_new = -h_g_hat[kx, :, kz]
                N_vort_old = -h_g_hat_old[kx, :, kz]
                phi_vort_hat_new = jnp.linalg.solve(
                    lhs_mat_vort,
                    rhs_mat_vort @ phi_vort_hat
                    + dt / 2 * (3 * N_vort_new - N_vort_old),
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
                    rhs_mat_ns = I_ + dt / (2 * Re) * L_NS
                    lhs_mat_ns = I_ - dt / (2 * Re) * L_NS
                    Nx = len(domain.grid[0])
                    Nz = len(domain.grid[2])
                    dx = Nx * (2 * jnp.pi / domain.scale_factors[0]) ** (2)
                    dz = Nz * (2 * jnp.pi / domain.scale_factors[2]) ** (2)
                    N_00_new = (
                        jnp.block(
                            [
                                -conv_ns_hat[0][kx__, :, kz__],
                                -conv_ns_hat[2][kx__, :, kz__],
                            ]
                        )
                        - dPdx
                        * (dx * dz) ** (1 / 2)
                        * domain.aliasing
                        * jnp.block(
                            [
                                jnp.ones(vel_hat[0][kx__, :, kz__].shape),
                                jnp.zeros(vel_hat[2][kx__, :, kz__].shape),
                            ]
                        )
                        - dPdz
                        * (dx * dz) ** (1 / 2)
                        * domain.aliasing
                        * jnp.block(
                            [
                                jnp.zeros(vel_hat[0][kx__, :, kz__].shape),
                                jnp.ones(vel_hat[2][kx__, :, kz__].shape),
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
                        - dPdx
                        * (dx * dz) ** (1 / 2)
                        * domain.aliasing
                        * jnp.block(
                            [
                                jnp.ones(vel_hat[0][kx__, :, kz__].shape),
                                jnp.zeros(vel_hat[2][kx__, :, kz__].shape),
                            ]
                        )
                        - dPdz
                        * (dx * dz) ** (1 / 2)
                        * domain.aliasing
                        * jnp.block(
                            [
                                jnp.zeros(vel_hat[0][kx__, :, kz__].shape),
                                jnp.ones(vel_hat[2][kx__, :, kz__].shape),
                            ]
                        )
                    )
                    v_hat_new = jnp.linalg.solve(
                        lhs_mat_ns,
                        rhs_mat_ns @ v_hat + dt / 2 * (3 * N_00_new - N_00_old),
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
                return (v_0_new_field, v_1_hat_new, v_2_new_field, v_1_lap_new_a)

            return fn

        v_1_hat_0 = vel_hat[1]
        v_1_lap_hat_0 = v_1_hat_0.laplacian()

        (
            h_v_hat_0,
            h_g_hat_0,
            vort_hat_0,
            conv_ns_hat_0,
        ) = update_nonlinear_terms_high_performance(
            self.domain_no_hat,
            jnp.array([vel_hat[0].field, vel_hat[1].field, vel_hat[2].field]),
            jnp.array(
                [vel_base_hat[0].field, vel_base_hat[1].field, vel_base_hat[2].field]
            ),
        )
        # old step (# TODO cache this)
        (
            h_v_hat_0_old,
            h_g_hat_0_old,
            _,
            conv_ns_hat_0_old,
        ) = update_nonlinear_terms_high_performance(
            self.domain_no_hat,
            jnp.array(
                [vel_hat_old[0].field, vel_hat_old[1].field, vel_hat_old[2].field]
            ),
            jnp.array(
                [vel_base_hat[0].field, vel_base_hat[1].field, vel_base_hat[2].field]
            ),
        )

        # solve equations
        vel_new_hat, _ = vel_hat.reconstruct_from_wavenumbers(
            perform_single_cn_ab_step_for_single_wavenumber(
                v_1_lap_hat_0,
                vort_hat_0,
                conv_ns_hat_0,
                conv_ns_hat_0_old,
                h_v_hat_0,
                h_g_hat_0,
                h_v_hat_0_old,
                h_g_hat_0_old,
            ),
            1,
        )
        vel_new_hat.update_boundary_conditions()

        vel_new_hat.name = "velocity_hat"
        for i in jnp.arange(len(vel_new_hat)):
            vel_new_hat[i].name = "velocity_hat_" + ["x", "y", "z"][i]
        self.append_field("velocity_hat", vel_new_hat)

    def perform_time_step(self):
        # return self.perform_runge_kutta_step()
        return self.perform_cn_ab_step()


def solve_navier_stokes_pertubation(
        Re=1.8e2, end_time=1e1, max_iter=1000, Nx=6, Ny=40, Nz=None, pertubation_factor=0.1, scale_factors=(1.87, 1.0, 0.93)
):
    Ny = Ny
    Nz = Nz or Nx + 4

    domain = Domain((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)
    # domain = Domain((Nx, Ny, Nz), (True, False, True))

    vel_x_fn = lambda X: (0.1
        * pertubation_factor
        * (
            jnp.pi
            / 3
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(3 * X[0])
            * jnp.cos(4 * X[2])
        ))
    # add small pertubation in y and z to see if it decays
    vel_y_fn = (
        lambda X: 0.1
        * pertubation_factor
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
        * pertubation_factor
        * (jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(5 * X[0]) * jnp.cos(3 * X[2]))
    )
    vel_x = Field.FromFunc(domain, vel_x_fn, name="vel_x")
    vel_y = Field.FromFunc(domain, vel_y_fn, name="vel_y")
    vel_z = Field.FromFunc(domain, vel_z_fn, name="vel_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")

    nse = NavierStokesVelVortPertubation.FromVelocityField((Nx, Ny, Nz), vel, Re)
    nse.end_time = end_time
    nse.max_iter = max_iter

    nse.after_time_step_fn = None

    return nse

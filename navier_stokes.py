#!/usr/bin/env python3

from types import NoneType
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from importlib import reload
import sys

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


class NavierStokesVelVort(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation)"

    def __init__(self, shape, *fields, **params):
        domain = Domain(shape, (True, False, True))
        super().__init__(domain, *fields)
        self.Re = params["Re"]
        self.flow_rate = self.get_flow_rate()

    @classmethod
    def FromVelocityField(cls, shape, velocity_field, Re=1.8e2):
        domain = Domain(shape, (True, False, True))
        velocity_field_hat = velocity_field.hat()
        velocity_field_hat.name = "velocity_hat"
        return cls(shape, velocity_field_hat, Re=Re)

    @classmethod
    def FromRandom(cls, shape, Re):
        domain = Domain(shape, (True, False, True))
        vel_x = Field.FromRandom(domain, name="u0")
        vel_y = Field.FromRandom(domain, name="u1")
        vel_z = Field.FromRandom(domain, name="u2")
        vel_y.update_boundary_conditions()
        vel = VectorField([vel_x, vel_y, vel_z], "velocity")
        return cls.FromVelocityField(shape, vel, Re)

    def get_vorticity_and_helicity(self):
        velocity_field_hat = self.get_latest_field("velocity_hat")
        vort_hat = velocity_field_hat.curl()
        for i in range(3):
            vort_hat[i].name = "vort_hat_" + str(i)

        hel_hat = velocity_field_hat.cross_product(vort_hat)
        for i in range(3):
            hel_hat[i].name = "hel_hat_" + str(i)
        return (vort_hat, hel_hat)

    def get_flow_rate(self):
        vel_hat = self.get_latest_field("velocity_hat")
        return vel_hat[0].no_hat().integrate(1, 1, 0.0).field[0, 0, 0]

    def get_cheb_mat_2_homogeneous_dirichlet(self):
        return self.get_initial_field("velocity_hat")[
            0
        ].get_cheb_mat_2_homogeneous_dirichlet(1)

    def get_rk_parameters(self):
        return (
            [29 / 96, -3 / 40, 1 / 6],
            [37 / 160, 5 / 24, 1 / 6],
            [8 / 15, 5 / 12, 3 / 4],
            [0, -17 / 60, -5 / 12],
        )

    def assemble_rk_matrices(self, Ly, dt, kx, kz, i):
        alpha, beta, _, _ = self.get_rk_parameters()
        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        n = D2_hom_diri.shape[0]
        Z = jnp.zeros((n, n))
        I = jnp.eye(2 * n)
        L = Ly + I * (-(kx**2 + kz**2))
        lhs_mat_inv = jnp.linalg.inv(I - beta[i] * dt * L)
        rhs_mat = I + alpha[i] * dt * L
        return (lhs_mat_inv, rhs_mat)

    def update_nonlinear_terms(self, vel_hat_new):
        vel_new = vel_hat_new.no_hat()
        vort_new = vel_new.curl()

        hel_new = vel_new.cross_product(vort_new)

        h_v_new = (
            -(hel_new[0].diff(0) + hel_new[2].diff(2)).diff(1)
            + hel_new[1].diff(0, 2)
            + hel_new[1].diff(2, 2)
        )
        h_g_new = hel_new[0].diff(2) - hel_new[2].diff(0)

        h_v_hat_new = h_v_new.hat()
        h_g_hat_new = h_g_new.hat()
        vort_hat_new = vort_new.hat()
        hel_hat_new = hel_new.hat()

        return (h_v_hat_new, h_g_hat_new, vort_hat_new, hel_hat_new)


    def perform_runge_kutta_step(self, dt, i):
        Re = self.Re
        vel_hat = self.get_latest_field("velocity_hat")

        # start runge-kutta stepping
        _, _, gamma, xi = self.get_rk_parameters()
        domain_y = Domain((len(self.domain.grid[1]),), (False,))

        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        n = D2_hom_diri.shape[0]
        Z = jnp.zeros((n, n))
        L = 1 / Re * jnp.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])


        # flow rate per unit thickness
        Nx = len(self.domain.grid[0])
        Nz = len(self.domain.grid[2])
        # TODO how to generalize this to the turbulent case? Why is the number of grid points important?
        dPdx = - (Nx * Nz)**(1/2) * self.flow_rate * 3/2 / Re
        dPdz = 0 # spanwise pressure gradient should be negligble
        D2 = self.get_cheb_mat_2_homogeneous_dirichlet()
        L_NS = 1 / Re * jnp.block([[D2, Z], [Z, D2]])


        def perform_single_rk_step_for_single_wavenumber(step, v_1_lap_hat, vort_hat, hel_hat, hel_hat_old, h_v_hat, h_g_hat, h_v_hat_old, h_g_hat_old):
            def fn(kx_, kz_):
                kx = int(kx_)
                kz = int(kz_)
                lhs_mat_inv, rhs_mat = self.assemble_rk_matrices(L, dt, kx, kz, step)

                vort_1_hat = vort_hat[1]
                phi_hat = jnp.block([v_1_lap_hat[kx, :, kz], vort_1_hat[kx, :, kz]])

                N_new = jnp.block([h_v_hat[kx, :, kz], h_g_hat[kx, :, kz]])
                N_old = jnp.block([h_v_hat_old[kx, :, kz], h_g_hat_old[kx, :, kz]])
                phi_hat_new = lhs_mat_inv @ (
                    rhs_mat @ phi_hat + (dt * gamma[step]) * N_new + (dt * xi[step]) * N_old
                )
                n = len(phi_hat) // 2
                v_1_lap_hat_new = FourierFieldSlice(
                    domain_y, 1, phi_hat_new[:n], "v_1_lap_hat_new", kx, kz
                )
                vort_1_hat_new = FourierFieldSlice(
                    domain_y, 1, phi_hat_new[n:], "vort_hat_new", kx, kz
                )

                # compute velocity in y direction
                v_1_new = v_1_lap_hat_new.solve_poisson()

                # compute velocities in x and z directions
                if kx == 0 and kz == 0:
                    lhs_mat_00_inv, rhs_mat_00 = self.assemble_rk_matrices(L_NS, dt, 0, 0, step)
                    v_hat = jnp.block([vel_hat[0][kx, :, kz], vel_hat[2][kx, :, kz]])
                    N_00_new = jnp.block([hel_hat[0][kx, :, kz], hel_hat[2][kx, :, kz]]) \
                        - dPdx * jnp.block([jnp.ones(vel_hat[0][kx, :, kz].shape), jnp.zeros(vel_hat[2][kx, :, kz].shape)]) \
                        - dPdz * jnp.block([jnp.zeros(vel_hat[0][kx, :, kz].shape), jnp.ones(vel_hat[2][kx, :, kz].shape)])
                    N_00_old = jnp.block([hel_hat_old[0][kx, :, kz], hel_hat_old[2][kx, :, kz]]) \
                        - dPdx * jnp.block([jnp.ones(vel_hat[0][kx, :, kz].shape), jnp.zeros(vel_hat[2][kx, :, kz].shape)]) \
                        - dPdz * jnp.block([jnp.zeros(vel_hat[0][kx, :, kz].shape), jnp.ones(vel_hat[2][kx, :, kz].shape)])
                    v_hat_new = lhs_mat_00_inv @ (
                    rhs_mat_00 @ v_hat + (dt * gamma[step]) * N_00_new + (dt * xi[step]) * N_00_old
                    )
                    v_0_new = FourierFieldSlice(domain_y, 1, v_hat_new[:n], "v_0_new", kx, kz)
                    v_2_new = FourierFieldSlice(domain_y, 1, v_hat_new[n:], "v_0_new", kx, kz)
                else:
                    minus_kx_kz_sq = -(kx**2 + kz**2)
                    v_0_new = (
                        -1j * kx * v_1_new.diff(0) + 1j * kz * vort_1_hat_new
                    ) / minus_kx_kz_sq
                    v_2_new = (
                        -1j * kz * v_1_new.diff(0) - 1j * kx * vort_1_hat_new
                    ) / minus_kx_kz_sq

                vel_hat_new = VectorField([v_0_new, v_1_new, v_2_new])
                vel_hat_new.name = "velocity_hat"
                for i in range(3):
                    vel_hat_new[i].name = "velocity_hat_" + str(i)

                return vel_hat_new
            return fn

        # perform first RK step
        v_1_hat_0 = vel_hat[1]
        v_1_lap_hat_0 = v_1_hat_0.laplacian()

        h_v_hat_0, h_g_hat_0, vort_hat_0, hel_hat_0 = self.update_nonlinear_terms(vel_hat)

        # solve equations
        vel_new_hat_1 = vel_hat.reconstruct_from_wavenumbers(perform_single_rk_step_for_single_wavenumber(0, v_1_lap_hat_0, vort_hat_0, hel_hat_0, hel_hat_0, h_v_hat_0, h_g_hat_0, h_v_hat_0, h_g_hat_0))
        vel_new_hat_1.update_boundary_conditions()
        # update nonlinear terms
        h_v_hat_1, h_g_hat_1, vort_hat_1, hel_hat_1 = self.update_nonlinear_terms(vel_new_hat_1)

        # perform second RK step
        v_1_hat_1 = vel_new_hat_1[1]
        v_1_lap_hat_1 = v_1_hat_1.laplacian()

        # solve equations
        vel_new_hat_2 = vel_hat.reconstruct_from_wavenumbers(perform_single_rk_step_for_single_wavenumber(1, v_1_lap_hat_1, vort_hat_1, hel_hat_1, hel_hat_0, h_v_hat_1, h_g_hat_1, h_v_hat_0, h_g_hat_0))
        vel_new_hat_2.update_boundary_conditions()
        # update nonlinear terms
        h_v_hat_2, h_g_hat_2, vort_hat_2, hel_hat_2 = self.update_nonlinear_terms(vel_new_hat_2)

        # perform third RK step
        v_1_hat_2 = vel_new_hat_2[1]
        v_1_lap_hat_2 = v_1_hat_2.laplacian()

        # solve equations
        vel_new_hat = vel_hat.reconstruct_from_wavenumbers(perform_single_rk_step_for_single_wavenumber(2, v_1_lap_hat_2, vort_hat_2, hel_hat_2, hel_hat_1, h_v_hat_2, h_g_hat_2, h_v_hat_1, h_g_hat_1))
        vel_new_hat.update_boundary_conditions()

        vel_new_hat.name = "velocity_hat"
        for i in range(len(vel_new_hat)):
            vel_new_hat[i].name = "velocity_hat_" + ["x", "y", "z"][i]
        self.append_field("velocity_hat", vel_new_hat)

    def perform_time_step(self, dt, i):
        return self.perform_runge_kutta_step(dt, i)


def solve_navier_stokes_3D_channel():
    start_time = time.time()
    Nx = 4
    Ny = Nx
    Nz = Nx

    Re = 1.8e2

    vel_x_fn = (
        lambda X: 0.1 * jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    )

    nse = NavierStokesVelVort.FromRandom((Nx, Ny, Nz), Re)
    nse.perform_runge_kutta_step(1e-5, 1)
    return

    vort = vel.curl()
    for i in range(3):
        vort[i].name = "vort_" + str(i)

    hel = vel.cross_product(vort)
    for i in range(3):
        hel[i].name = "hel_" + str(i)

    vy_lap = vel[1].laplacian()
    vy_lap.name = "vy_lap_" + str(0)

    vort_1 = vort[1]

    h_v = -(hel[0].diff(0) + hel[2].diff(2)).diff(1) + hel[1].laplacian()
    h_g = hel[0].diff(2) - hel[2].diff(0)

    Nt = 500
    vy_laps = [vy_lap]
    vort_1_s = [vort_1]
    dt = 5e-5
    print(
        "Starting time loop, time for preparation: "
        + str(time.time() - start_time)
        + " seconds"
    )
    for i in range(1, Nt + 1):
        vy_laps.append(
            vy_laps[-1].perform_time_step(
                -h_v + vy_laps[-1].laplacian() * (1 / Re), dt, i
            )
        )
        vort_1_s.append(
            vort_1_s[-1].perform_time_step(
                -h_g + vort_1_s[-1].laplacian() * (1 / Re), dt, i
            )
        )

        vy_laps[-1].name = "vy_lap_" + str(i)
        vort_1_s[-1].name = "vort_1_" + str(i)

        # TODO vel is not being updated
        vort = vel.curl()
        hel = vel.cross_product(vort)

        h_v = -(hel[0].diff(0) + hel[2].diff(2)).diff(1) + hel[1].laplacian()
        h_g = hel[0].diff(2) - hel[2].diff(0)

    # u_final = jnp.fft.ifft(us_hat[-1], axis=0)
    # u_final = us[-1]
    # vy_lap.plot(vy_laps[-1])
    # vort_1.plot(vort_1_s[-1])

    end_time = time.time()
    print("elapsed time: " + str(end_time - start_time) + " seconds")

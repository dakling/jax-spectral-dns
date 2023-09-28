#!/usr/bin/env python3

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
        # vort = self.fields["vort"][0]
        # vort.update_boundary_conditions()

    @classmethod
    def FromVelocityField(cls, shape, velocity_field, Re=1.8e2):
        domain = Domain(shape, (True, False, True))
        vort = velocity_field.curl()
        for i in range(3):
            vort[i].name = "vort_" + str(i)

        hel = velocity_field.cross_product(vort)
        for i in range(3):
            hel[i].name = "hel_" + str(i)

        # return cls(domain, velocity_field, vort, hel, Re=Re)
        return cls(shape, velocity_field, Re=Re)

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
        velocity_field = self.get_latest_field("velocity")
        vort = velocity_field.curl()
        for i in range(3):
            vort[i].name = "vort_" + str(i)

        hel = velocity_field.cross_product(vort)
        for i in range(3):
            hel[i].name = "hel_" + str(i)
        return (vort, hel)

    def get_cheb_mat_2_homogeneous_dirichlet(self):
        return self.get_initial_field("velocity")[
            0
        ].get_cheb_mat_2_homogeneous_dirichlet(1)

    def get_rk_parameters(self):
        return (
            [29 / 96, -3 / 40, 1 / 6],
            [37 / 160, 5 / 24, 1 / 6],
            [8 / 15, 5 / 12, 3 / 4],
            [-17 / 60, -5 / 12],
        )

    def update_nonlinear_terms(self, domain_y, phi_hat, kx, kz):
        n = len(phi_hat)//2
        v_1_lap_hat_new = FourierFieldSlice(
            domain_y, 1, phi_hat[:n], "v_1_lap_hat_new", kx, kz
        )
        vort_1_hat_new = FourierFieldSlice(
            domain_y, 1, phi_hat[n:], "vort_hat_new", kx, kz
        )

        # v_1_new = v_1_lap_hat_new.integrate(1, 2, 0.0, 0.0)
        v_1_new = v_1_lap_hat_new.solve_poisson()
        minus_kx_kz_sq = -(kx**2 + kz**2)
        v_0_new = (
            -1j * kx * v_1_new.diff(0) + 1j * kz * vort_1_hat_new
        ) / minus_kx_kz_sq
        v_2_new = (
            -1j * kz * v_1_new.diff(0) - 1j * kx * vort_1_hat_new
        ) / minus_kx_kz_sq
        vel_new = VectorField([v_0_new, v_1_new, v_2_new])
        vel_new.name = "velocity"
        for i in range(3):
            vel_new[i].name = "velocity_" + str(i)

        vort_hat_new = vel_new.curl()

        hel_hat_new = vel_new.cross_product(vort_hat_new)

        h_v_hat_new = (
            -(hel_hat_new[0].diff(0) + hel_hat_new[2].diff(2)).diff(1) + hel_hat_new[1].laplacian()
        )
        h_g_hat_new = hel_hat_new[0].diff(2) - hel_hat_new[2].diff(0)

        # phi_hat_new = jnp.block([v_1_lap_hat_new.field, vort_1_hat_new.field])
        return (h_v_hat_new, h_g_hat_new, vel_new)

    def perform_runge_kutta_step(self, dt, i):
        Re = self.Re
        vel = self.get_latest_field("velocity")

        v_1_hat = vel[1].hat()
        v_1_lap_hat = v_1_hat.laplacian()

        vort, hel = self.get_vorticity_and_helicity()

        vort_hat = vort.hat()
        vort_1_hat = vort_hat[1]
        hel_hat = hel.hat()

        h_v_hat = (
            -(hel_hat[0].diff(0) + hel_hat[2].diff(2)).diff(1) + hel_hat[1].laplacian()
        )
        h_g_hat = hel_hat[0].diff(2) - hel_hat[2].diff(0)

        # start runge-kutta stepping
        alpha, beta, gamma, xi = self.get_rk_parameters()
        domain_hat = self.domain.hat()

        domain_y = Domain((len(self.domain.grid[1]),), (False,))

        D2_hom_diri = self.get_cheb_mat_2_homogeneous_dirichlet()
        n = D2_hom_diri.shape[0]
        Z = jnp.zeros((n, n))
        L = 1 / Re * jnp.block([[D2_hom_diri, Z], [Z, D2_hom_diri]])
        I = jnp.eye(2 * n)
        lhs_mat_inv_0 = jnp.linalg.inv(I - beta[0] * dt * L)
        rhs_mat_0 = I + alpha[0] * dt * L

        lhs_mat_inv_1 = jnp.linalg.inv(I - beta[1] * dt * L)
        rhs_mat_1 = I + alpha[1] * dt * L

        lhs_mat_inv_2 = jnp.linalg.inv(I - beta[2] * dt * L)
        rhs_mat_2 = I + alpha[2] * dt * L

        for kx_ in domain_hat.grid[0][1:]:
            for kz_ in domain_hat.grid[2][1:]:
                kx = int(kx_)
                kz = int(kz_)
                # first RK step
                phi_hat = jnp.block([v_1_lap_hat[kx, :, kz], vort_1_hat[kx, :, kz]])
                N_0 = jnp.block([h_v_hat[kx, :, kz], h_g_hat[kx, :, kz]])
                phi_hat_new_1 = lhs_mat_inv_0 @ (
                    rhs_mat_0 @ phi_hat + (dt * gamma[0]) * N_0
                )

                # update nonlinear terms
                # h_v_hat_new_1, h_g_hat_new_1, _ = self.update_nonlinear_terms(domain_y, phi_hat_new_1, kx, kz)
                h_v_hat_new_1, h_g_hat_new_1, vel_new_1 = self.update_nonlinear_terms(domain_y, phi_hat_new_1, kx, kz)

                # second RK step
                N_1 = jnp.block([h_v_hat_new_1.field, h_g_hat_new_1.field])
                phi_hat_new_2 = lhs_mat_inv_1 @ (
                    rhs_mat_1 @ phi_hat_new_1 + (dt * gamma[1]) * N_1 + xi[0] * N_0
                )

                v_curl = vel_new_1.curl()
                hel_ = vel_new_1.cross_product(v_curl)
                # vel_new_1.plot(v_curl[0], v_curl[1], v_curl[2], hel_[0], hel_[1], hel_[2])
                vel_new_1[1].plot(FourierFieldSlice(domain_y, 1, vel[1].hat()[kx, :, kz], "v0", kx, kz))
                print(jnp.linalg.norm(N_1))
                fig, ax = plt.subplots(1,1)
                ax.plot(domain_y.grid[0], phi_hat_new_2[:n])
                ax.plot(domain_y.grid[0], phi_hat_new_2[n:])
                # ax.plot(domain_y.grid[0], phi_hat[:n], "--")
                # ax.plot(domain_y.grid[0], phi_hat[n:], "--")
                fig.savefig("plots/plot.pdf")
                return
                # update nonlinear terms
                h_v_hat_new_2, h_g_hat_new_2, _ = self.update_nonlinear_terms(domain_y, phi_hat_new_2, kx, kz)

                # third RK step
                N_2 = jnp.block([h_v_hat_new_2.field, h_g_hat_new_2.field])
                phi_hat_new_3 = lhs_mat_inv_2 @ (
                    rhs_mat_2 @ phi_hat_new_2 + (dt * gamma[2]) * N_2 + xi[1] * N_1
                )

                # update velocity
                _, _, vel_new = self.update_nonlinear_terms(domain_y, phi_hat_new_3, kx, kz)
                # TODO cast this back into non-sliced form
                # self.fields["velocity"].append(vel_new)
                # return vel_new

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
    vy_lap.plot(vy_laps[-1])
    vort_1.plot(vort_1_s[-1])

    end_time = time.time()
    print("elapsed time: " + str(end_time - start_time) + " seconds")

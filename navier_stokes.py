#!/usr/bin/env python3

import jax.numpy as jnp
import time

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
from field import Field, VectorField

try:
    reload(sys.modules["equation"])
except:
    pass
from equation import Equation


class NavierStokesVelVort(Equation):
    name = "Navier Stokes equation (velocity-vorticity formulation)"

    def __init__(self, domain, *fields, **params):
        super().__init__(domain, *fields)
        self.Re = params["Re"]
        # vort = self.fields["vort"][0]
        # vort.update_boundary_conditions()

    @classmethod
    def FromVelocityField(cls, domain, velocity_field, Re=1.8e2):
        vort = velocity_field.curl()
        for i in range(3):
            vort[i].name = "vort_" + str(i)

        hel = velocity_field.cross_product(vort)
        for i in range(3):
            hel[i].name = "hel_" + str(i)

        # return cls(domain, velocity_field, vort, hel, Re=Re)
        return cls(domain, velocity_field, Re=Re)

    @classmethod
    def FromRandom(cls, domain, Re):
        vel_x = Field.FromRandom(domain, name="u0")
        vel_y = Field.FromRandom(domain, name="u1")
        vel_z = Field.FromRandom(domain, name="u2")
        vel_y.update_boundary_conditions()
        vel = VectorField([vel_x, vel_y, vel_z], "velocity")
        return cls.FromVelocityField(domain, vel, Re)

    def get_vorticity_and_helicity(self):
        velocity_field = self.get_latest_field("velocity")
        vort = velocity_field.curl()
        for i in range(3):
            vort[i].name = "vort_" + str(i)

        hel = velocity_field.cross_product(vort)
        for i in range(3):
            hel[i].name = "hel_" + str(i)
        return (vort, hel)

    def get_rk_parameters(self):
        return ([0], [0], [0]) # TODO

    def perform_runge_kutta_step(self, dt, i):
        Re = self.Re
        vel = self.get_latest_field("velocity")

        v_1_hat = vel[1].hat()
        v_1_lap_hat = v_1_hat.laplacian()

        vort, hel = self.get_vorticity_and_helicity()

        vort_hat = vort.hat()
        vort_1_hat = vort_hat[1]
        hel_hat = hel.hat()

        # h_v = -(hel[0].diff(0) + hel[2].diff(2)).diff(1) + hel[1].laplacian()
        # h_g = hel[0].diff(2) - hel[2].diff(0)

        h_v_hat = -(hel_hat[0].diff(0) + hel_hat[2].diff(2)).diff(1) + hel_hat[1].laplacian()
        h_g_hat = hel_hat[0].diff(2) - hel_hat[2].diff(0)

        # start runge-kutta stepping
        alpha, beta, gamma = self.get_rk_parameters()
        domain_hat = self.domain.hat()
        for kx in domain_hat.grid[0]:
            for kz in domain_hat.grid[2]:
                phi_hat = jnp.block([v_1_lap_hat[kx, kz], vort_1_hat[kx, kz]])
        v_1_lap_hat_new_1 = (1 / (1 - dt * beta[0] ))((1 + dt * alpha[0]) * (1.0/Re * v_1_lap_hat.laplacian()) + (dt * gamma[0]) * h_v_hat)



    def perform_time_step(self, dt, i):
        return self.perform_runge_kutta_step(dt, i)


def solve_navier_stokes_3D_channel():
    start_time = time.time()
    Nx = 24
    Ny = Nx
    Nz = Nx

    Re = 1.8e2

    domain = Domain((Nx, Ny, Nz), (True, False, True))

    vel_x_fn = (
        lambda X: 0.1 * jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    )

    nse = NavierStokesVelVort.FromRandom(domain, Re)
    print(nse)
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

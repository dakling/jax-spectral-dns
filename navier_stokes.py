#!/usr/bin/env python3

from types import NoneType
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

import numpy as np

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

def perform_channel_simulation_cheb_fourier_3D():
    Nx = 24
    Ny = Nx
    Nz = Nx

    Re = 1.8e2

    domain = Domain((Nx, Ny, Nz), (True, False, True))

    vel_x_fn = lambda X: 0.1 * jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi/2)

    vel_x = Field.FromFunc(domain, func=vel_x_fn, name="u0")
    vel_y = Field.FromFunc(domain, name="u1")
    vel_z = Field.FromFunc(domain, name="u2")

    vel_y.update_boundary_conditions()

    vel = VectorField([vel_x, vel_y, vel_z])

    # return
    vort = vel.curl()
    for i in range(3):
        vort[i].name = "vort_" + str(i)

    hel = vel.cross_product(vort)
    for i in range(3):
        hel[i].name = "hel_" + str(i)

    vy_lap = vel[1].laplacian()

    vort_1 = vort[1]

    h_v = - (hel[0].diff(0) + hel[2].diff(2)).diff(1) + hel[1].laplacian()
    h_g = hel[0].diff(2) - hel[2].diff(0)

    Nt = 500
    vy_laps = [vy_lap]
    vort_1_s = [vort_1]
    dt = 5e-5
    for i in range(1,Nt+1):
        vy_laps.append(vy_laps[-1].perform_time_step(-h_v + vy_laps[-1].laplacian() * (1/Re) , dt, i))
        vort_1_s.append(vort_1_s[-1].perform_time_step(-h_g + vort_1_s[-1].laplacian() * (1/Re), dt, i))

        vy_laps[-1].name = "vy_lap_" + str(i)
        vort_1_s[-1].name = "vort_1_" + str(i)

        vort = vel.curl()
        hel = vel.cross_product(vort)

        h_v = - (hel[0].diff(0) + hel[2].diff(2)).diff(1) + hel[1].laplacian()
        h_g = hel[0].diff(2) - hel[2].diff(0)

    # u_final = jnp.fft.ifft(us_hat[-1], axis=0)
    # u_final = us[-1]
    vy_lap.plot(vy_laps[-1])
    vort_1.plot(vort_1_s[-1])

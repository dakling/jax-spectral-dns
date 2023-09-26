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

    domain = Domain((Nx, Ny, Nz), (True, False, True))

    vel_x_fn = lambda X: 0.1 * jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi/2)

    vel_x = Field.FromFunc(domain, func=vel_x_fn, name="u1")
    vel_y = Field.FromFunc(domain, name="u2")
    vel_z = Field.FromFunc(domain, name="u3")

    vel_y.update_boundary_conditions()

    vel = VectorField([vel_x, vel_y, vel_z])
    vel.plot()

    # return
    vort = vel.curl()
    for i in range(3):
        vort[i].name = "vort_" + str(i)
    vort.plot()

    hel = vel.cross_product(vort)
    for i in range(3):
        hel[i].name = "hel_" + str(i)
    hel.plot()
    return
    # vort.update_boundary_conditions()
    u_xx = u.diff(0, order=2)
    u_yy = u.diff(1, order=2)

    Nt = 5000
    # us_hat = [u_hat]
    us = [u]
    dt = 5e-5
    for i in range(1,Nt+1):
        us.append(us[-1].perform_time_step(u_xx + u_yy, dt, i))
        u_xx = us[-1].diff(0, order=2)
        u_yy = us[-1].diff(1, order=2)
    # u_final = jnp.fft.ifft(us_hat[-1], axis=0)
    # u_final = us[-1]
    u.plot(us[-1])

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
from field import Field

def perform_simulation_cheb_fourier_2D_no_mat():
    Nx = 24
    Ny = Nx

    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi/2)
    # u = Field(domain, func=u_fn, name="u")
    u = Field.FromFunc(domain, func=u_fn, name="u_0")

    # u.update_boundary_conditions() # TODO direction
    # return

    # u_hat = u.hat()
    # u_xx_hat = u_hat.diff(0, order=2)
    # return
    # u_yy_hat = (cheb_mat_2 @ u_hat.T).T
    # u_yy_hat = u_hat.diff(1, order=2)

    # u_x = four_mat_1 @ u
    # u_xx = four_mat_2 @ u
    # u_y = (cheb_mat_1 @ u.T).T
    # u_yy = (cheb_mat_2 @ u.T).T
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

#!/usr/bin/env python3

import jax.numpy as jnp

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

def perform_simulation_cheb_fourier_2D():
    Nx = 24
    Ny = Nx

    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi/2)
    u = Field.FromFunc(domain, func=u_fn, name="u_0")

    u.update_boundary_conditions()
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

def perform_simulation_cheb_fourier_3D():
    Nx = 24
    Ny = Nx
    Nz = Nx

    domain = Domain((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi/2)
    u = Field.FromFunc(domain, func=u_fn, name="u_0")

    u.update_boundary_conditions()
    u_xx = u.diff(0, order=2)
    u_yy = u.diff(1, order=2)
    u_zz = u.diff(2, order=2)

    Nt = 5000
    us = [u]
    dt = 5e-5
    for i in range(1,Nt+1):
        us.append(us[-1].perform_time_step(u_xx + u_yy + u_zz, dt, i))
        u_xx = us[-1].diff(0, order=2)
        u_yy = us[-1].diff(1, order=2)
    u.plot_center(0, us[-1])
    u.plot_center(1, us[-1])
    u.plot_center(2, us[-1])

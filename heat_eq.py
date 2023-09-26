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

def test():
    Nx = 24
    Ny = Nx
    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi/2)
    u = Field.FromFunc(domain, func=u_fn, name="u")
    print(u)
    # print(u.field[:, 0]) # the same as...
    # print(u.field.take(0, axis=1))
    u_x = u.diff(0, 1)
    # print(u_x)
    u_xx = u.diff(0, 2)
    u_y = u.diff(1, 1)
    u_yy = u.diff(1, 2)

    u_x_ana = Field.FromFunc(domain, func=lambda X: - jnp.sin(X[0]) * jnp.cos(X[1] * jnp.pi/2), name="u_x_ana")
    u_xx_ana = Field.FromFunc(domain, func=lambda X: - jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi/2), name="u_xx_ana")
    u_y_ana = Field.FromFunc(domain, func=lambda X: - jnp.cos(X[0]) * jnp.pi/2 * jnp.sin(X[1] * jnp.pi/2), name="u_y_ana")
    u_yy_ana = Field.FromFunc(domain, func=lambda X: - jnp.cos(X[0]) * (jnp.pi/2)**2 * jnp.cos(X[1] * jnp.pi/2), name="u_yy_ana")

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana, u_x - u_x_ana)
    u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
    tol = 1e-5
    print(abs(u_x - u_x_ana))
    print(abs(u_xx - u_xx_ana))
    print(abs(u_y - u_y_ana) )
    print(abs(u_yy - u_yy_ana) )
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol
    assert abs(u_y - u_y_ana) < tol
    assert abs(u_yy - u_yy_ana) < tol

def perform_simulation_cheb_fourier_2D_no_mat():
    Nx = 4
    Ny = Nx

    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi/2)
    # u = Field(domain, func=u_fn, name="u")
    u = Field.FromFunc(domain, func=u_fn, name="u")

    test()
    # print(u)
    # u.update_boundary_conditions() # TODO direction
    # print(u)
    return

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
    for _ in range(Nt):
        new_u = us[-1] + (u_xx + u_yy) * dt
        new_u_bc = (pad_mat_with_zeros(new_u[:,1:-1])[1:-1,:])
        # new_u_hat = us_hat[-1] + (u_xx_hat + u_yy_hat) * dt
        # new_u_hat_bc = (pad_mat_with_zeros(new_u_hat[:,1:-1])[1:-1,:])
        us.append(new_u_bc)
        # us_hat.append(new_u_hat_bc)
        # u_xx_hat = jnp.linalg.matrix_power((1j*jnp.diag(ks)), 2) @ new_u_hat_bc
        # u_yy_hat = (cheb_mat_2 @ new_u_hat_bc.T).T
        u_xx = (four_mat_2 @ new_u_bc)
        u_yy = (cheb_mat_2 @ new_u_bc.T).T
    # u_final = jnp.fft.ifft(us_hat[-1], axis=0)
    u_final = us[-1]
    fig, ax = plt.subplots(1,2)
    ax[0].plot(xs, u[:, Ny//2])
    # ax[0].plot(xs, u_x[:, Ny//2])
    # ax[0].plot(xs, u_xx[:, Ny//2])
    ax[1].plot(ys, u[Nx//2, :])
    # ax[1].plot(ys, u_y[Nx//2, :])
    # ax[1].plot(ys, u_yy[Nx//2, :])
    ax[0].plot(xs, u_final[:, Ny//2])
    ax[1].plot(ys, u_final[Nx//2, :])
    fig.savefig("plot.pdf")

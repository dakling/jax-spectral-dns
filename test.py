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


def test_1D_cheb():
    Nx = 48
    domain = Domain((Nx,), (False,))

    u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u_1d_cheb")
    u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)

    u_x_ana = Field.FromFunc(
        domain, func=lambda X: -jnp.sin(X[0] * jnp.pi / 2) * jnp.pi / 2, name="u_x_ana"
    )
    u_xx_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * jnp.pi / 2) * (jnp.pi / 2) ** 2,
        name="u_xx_ana",
    )

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    tol = 5e-4
    print(abs(u_x - u_x_ana))
    print(abs(u_xx - u_xx_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol


def test_1D_periodic():
    Nx = 24
    domain = Domain((Nx,), (True,))

    u_fn = lambda X: jnp.cos(X[0])
    u = Field.FromFunc(domain, func=u_fn, name="u_1d_periodic")
    u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)

    u_x_ana = Field.FromFunc(domain, func=lambda X: -jnp.sin(X[0]), name="u_x_ana")
    u_xx_ana = Field.FromFunc(domain, func=lambda X: -jnp.cos(X[0]), name="u_xx_ana")

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    tol = 5e-5
    print(abs(u_x - u_x_ana))
    print(abs(u_xx - u_xx_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol


def test_2D():
    Nx = 24
    Ny = Nx
    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u_2d")
    u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)
    u_y = u.diff(1, 1)
    u_yy = u.diff(1, 2)

    u_x_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.sin(X[0]) * jnp.cos(X[1] * jnp.pi / 2),
        name="u_x_ana",
    )
    u_xx_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2),
        name="u_xx_ana",
    )
    u_y_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * jnp.pi / 2 * jnp.sin(X[1] * jnp.pi / 2),
        name="u_y_ana",
    )
    u_yy_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * (jnp.pi / 2) ** 2 * jnp.cos(X[1] * jnp.pi / 2),
        name="u_yy_ana",
    )

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
    tol = 5e-5
    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
    # print(abs(u_y - u_y_ana))
    # print(abs(u_yy - u_yy_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol
    assert abs(u_y - u_y_ana) < tol
    assert abs(u_yy - u_yy_ana) < tol


def test_3D():
    Nx = 48
    Ny = Nx
    Nz = Nx
    domain = Domain((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(X[2])
    u = Field.FromFunc(domain, func=u_fn, name="u_3d")
    # u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)
    u_y = u.diff(1, 1)
    u_yy = u.diff(1, 2)
    u_z = u.diff(2, 1)
    u_zz = u.diff(2, 2)

    u_x_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.sin(X[0]) * jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(X[2]),
        name="u_x_ana",
    )
    u_xx_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(X[2]),
        name="u_xx_ana",
    )
    u_y_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0])
        * jnp.pi
        / 2
        * jnp.sin(X[1] * jnp.pi / 2)
        * jnp.cos(X[2]),
        name="u_y_ana",
    )
    u_yy_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0])
        * (jnp.pi / 2) ** 2
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2]),
        name="u_yy_ana",
    )
    u_z_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2) * jnp.sin(X[2]),
        name="u_z_ana",
    )
    u_zz_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2) * jnp.cos(X[2]),
        name="u_zz_ana",
    )

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
    u.plot_center(2, u_z, u_zz, u_z_ana, u_zz_ana)

    print(abs(u_x - u_x_ana))
    print(abs(u_xx - u_xx_ana))
    print(abs(u_y - u_y_ana))
    print(abs(u_yy - u_yy_ana))
    print(abs(u_z - u_z_ana))
    print(abs(u_zz - u_zz_ana))
    tol = 5e-5
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol
    assert abs(u_y - u_y_ana) < tol
    assert abs(u_yy - u_yy_ana) < tol
    assert abs(u_z - u_z_ana) < tol
    assert abs(u_zz - u_zz_ana) < tol


# def test_integration():
#     Nx = 24
#     # domain = Domain((Nx,), (False,))
#     domain = Domain((Nx,), (True,))

#     # u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
#     u_fn = lambda X: jnp.cos(X[0])
#     u = Field.FromFunc(domain, func=u_fn, name="u_1d")
#     # u_int = Field(domain, u.solve_poisson([0], u))
#     u_int = u.solve_poisson([0], u)
#     # u_int.update_boundary_conditions()
#     u_int.name="u_int"
#     u_int.plot(u)

def test_fourier_1D():

    Nx = 24
    domain = Domain((Nx,), (True,))

    u_fn = lambda X: jnp.cos(X[0])
    u = Field.FromFunc(domain, func=u_fn, name="u_1d")
    u_hat = u.hat()

    u_diff_fn = lambda X: -jnp.sin(X[0])
    u_diff_ana = Field.FromFunc(domain, func=u_diff_fn, name="u_1d_diff_ana")

    u_diff_fn_2 = lambda X: -jnp.cos(X[0])
    u_diff_ana_2 = Field.FromFunc(domain, func=u_diff_fn_2, name="u_1d_diff_2_ana")

    u_int_fn = lambda X: jnp.sin(X[0])
    u_int_ana = Field.FromFunc(domain, func=u_int_fn, name="u_1d_int_ana")

    u_int_fn_2 = lambda X: -jnp.cos(X[0])
    u_int_ana_2 = Field.FromFunc(domain, func=u_int_fn_2, name="u_1d_int_2_ana")

    u_hat_int = u_hat.integrate()
    u_int = u_hat_int.no_hat()
    u_int.name = "u_1d_int"
    u_hat_int_2 = u_hat.integrate(2)
    u_int_2 = u_hat_int_2.no_hat()
    u_int_2.name = "u_1d_int_2"
    tol = 1e-6
    # print(abs(u_int - u_int_ana))
    # print(abs(u_int_2 - u_int_ana_2))
    assert abs(u_int - u_int_ana) < tol
    assert abs(u_int_2 - u_int_ana_2) < tol

    u_hat_diff = u_hat.diff(0, 1)
    u_hat_diff_2 = u_hat.diff(0, 2)
    u_diff = u_hat_diff.no_hat()
    u_diff.name = "u_1d_diff"
    u_diff_2 = u_hat_diff_2.no_hat()
    u_diff_2.name = "u_1d_diff_2"
    u.plot(u_diff, u_diff_2, u_int, u_int_2)
    # print(abs(u_diff - u_diff_ana))
    # print(abs(u_diff_2 - u_diff_ana_2))
    assert abs(u_diff - u_diff_ana) < tol
    assert abs(u_diff_2 - u_diff_ana_2) < tol

def run_all_tests():
    # test_1D_periodic()
    # test_1D_cheb()
    # test_2D()
    # test_3D()
    test_fourier_1D()

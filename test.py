#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.figure as figure
from matplotlib import legend
from numpy import float128
import scipy as sc
import scipy.optimize as optimization
import time

from cProfile import Profile
from pstats import SortKey, Stats

# import numpy as np

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    print("Unable to load Domain")
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    print("Unable to load Field")
from field import Field, FourierFieldSlice, VectorField

try:
    reload(sys.modules["navier_stokes"])
except:
    print("Unable to load Navier Stokes")
from navier_stokes import NavierStokesVelVort, solve_navier_stokes_laminar

try:
    reload(sys.modules["navier_stokes_pertubation"])
except:
    print("Unable to load navier-stokes-pertubation")
from navier_stokes_pertubation import solve_navier_stokes_pertubation

try:
    reload(sys.modules["linear_stability_calculation"])
except:
    print("Unable to load linear stability")
from linear_stability_calculation import LinearStabilityCalculation

NoneType = type(None)


def test_1D_cheb():
    Nx = 48
    domain = Domain((Nx,), (False,))

    u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    # u_fn = lambda X: 0.0 * jnp.cos(X[0] * jnp.pi / 2) + 2
    u = Field.FromFunc(domain, func=u_fn, name="u_1d_cheb")
    # u.update_boundary_conditions()
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

    # u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    # print("u")
    # print(u)
    # print("ux")
    # print(u_x)
    u.plot_center(0, u_x, u_xx)
    tol = 5e-4
    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol


def test_1D_periodic():
    Nx = 24
    scale_factor = 1.0
    domain = Domain((Nx,), (True,), scale_factors=(scale_factor,))

    u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
    u = Field.FromFunc(domain, func=u_fn, name="u_1d_periodic")
    u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)

    u_diff_fn = (
        lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor) * 2 * jnp.pi / scale_factor
    )
    u_diff_fn_2 = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        * (2 * jnp.pi / scale_factor) ** 2
    )
    u_x_ana = Field.FromFunc(domain, func=u_diff_fn, name="u_x_ana")
    u_xx_ana = Field.FromFunc(domain, func=u_diff_fn_2, name="u_xx_ana")

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    tol = 8e-5
    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol


def test_2D():
    Nx = 20
    # Ny = Nx
    Ny = 24
    scale_factor = 1.0
    domain = Domain((Nx, Ny), (True, False), scale_factors=(scale_factor, 1.0))

    u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor) * jnp.cos(
        X[1] * jnp.pi / 2
    )
    u = Field.FromFunc(domain, func=u_fn, name="u_2d")
    u.update_boundary_conditions()
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)
    u_y = u.diff(1, 1)
    u_yy = u.diff(1, 2)

    u_x_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor)
        * jnp.cos(X[1] * jnp.pi / 2)
        * 2
        * jnp.pi
        / scale_factor,
        name="u_x_ana",
    )
    u_xx_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        * jnp.cos(X[1] * jnp.pi / 2)
        * (2 * jnp.pi / scale_factor) ** 2,
        name="u_xx_ana",
    )
    u_y_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        * jnp.pi
        / 2
        * jnp.sin(X[1] * jnp.pi / 2),
        name="u_y_ana",
    )
    u_yy_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        * (jnp.pi / 2) ** 2
        * jnp.cos(X[1] * jnp.pi / 2),
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
    Nx = 24
    Ny = 40
    Nz = 20
    scale_factor_x = 1.0
    scale_factor_z = 2.0
    domain = Domain(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(scale_factor_x, 1.0, scale_factor_z),
    )

    u_fn = (
        lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
    )
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
        func=lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
        * 2
        * jnp.pi
        / scale_factor_x,
        name="u_x_ana",
    )
    u_xx_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
        * (2 * jnp.pi / scale_factor_x) ** 2,
        name="u_xx_ana",
    )
    u_y_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.pi
        / 2
        * jnp.sin(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z),
        name="u_y_ana",
    )
    u_yy_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * (jnp.pi / 2) ** 2
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z),
        name="u_yy_ana",
    )
    u_z_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.sin(X[2] * 2 * jnp.pi / scale_factor_z)
        * 2
        * jnp.pi
        / scale_factor_z,
        name="u_z_ana",
    )
    u_zz_ana = Field.FromFunc(
        domain,
        func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
        * (2 * jnp.pi / scale_factor_z) ** 2,
        name="u_zz_ana",
    )

    u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
    u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
    u.plot_center(2, u_z, u_zz, u_z_ana, u_zz_ana)

    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
    # print(abs(u_y - u_y_ana))
    # print(abs(u_yy - u_yy_ana))
    # print(abs(u_z - u_z_ana))
    # print(abs(u_zz - u_zz_ana))
    tol = 5e-5
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol
    assert abs(u_y - u_y_ana) < tol
    assert abs(u_yy - u_yy_ana) < tol
    assert abs(u_z - u_z_ana) < tol
    assert abs(u_zz - u_zz_ana) < tol


def test_fourier_1D():
    Nx = 24
    scale_factor = 1.0
    # scale_factor = 2 * jnp.pi
    domain = Domain((Nx,), (True,), scale_factors=(scale_factor,))
    # domain = Domain((Nx,), (True,))

    u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
    # u_fn = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / scale_factor))
    u = Field.FromFunc(domain, func=u_fn, name="u_1d")
    u_hat = u.hat()

    u_diff_fn = (
        lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor) * 2 * jnp.pi / scale_factor
        # lambda X: jnp.cos(X[0]) / scale_factor * jnp.exp(jnp.sin(X[0] / scale_factor))
    )
    u_diff_ana = Field.FromFunc(domain, func=u_diff_fn, name="u_1d_diff_ana")

    u_diff_fn_2 = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        * (2 * jnp.pi / scale_factor) ** 2
    )
    u_diff_ana_2 = Field.FromFunc(domain, func=u_diff_fn_2, name="u_1d_diff_2_ana")

    u_int_fn = lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor) * (
        2 * jnp.pi / scale_factor
    ) ** (-1)
    u_int_ana = Field.FromFunc(domain, func=u_int_fn, name="u_1d_int_ana")

    u_int_fn_2 = lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor) * (
        2 * jnp.pi / scale_factor
    ) ** (-2)
    u_int_ana_2 = Field.FromFunc(domain, func=u_int_fn_2, name="u_1d_int_2_ana")

    u_hat_int = u_hat.integrate(0)
    u_int = u_hat_int.no_hat()
    u_int.name = "u_1d_int"
    u_hat_int_2 = u_hat.integrate(0, 2)
    u_int_2 = u_hat_int_2.no_hat()
    u_int_2.name = "u_1d_int_2"
    tol = 7e-5

    u_hat_diff = u_hat.diff(0, 1)
    u_hat_diff_2 = u_hat.diff(0, 2)
    u_diff = u_hat_diff.no_hat()
    u_diff.name = "u_1d_diff"
    u_diff_2 = u_hat_diff_2.no_hat()
    u_diff_2.name = "u_1d_diff_2"
    u_int.plot(u)
    # u.integrate(0).plot(u)
    # print(abs(u_int - u_int_ana))
    # print(abs(u_int_2 - u_int_ana_2))
    # print(abs(u_diff - u_diff_ana))
    # print(abs(u_diff_2 - u_diff_ana_2))
    assert abs(u_int - u_int_ana) < tol
    assert abs(u_int_2 - u_int_ana_2) < tol
    assert abs(u_diff - u_diff_ana) < tol
    assert abs(u_diff_2 - u_diff_ana_2) < tol


def test_fourier_2D():
    Nx = 24
    Ny = Nx + 4
    scale_factor_x = 1.0
    scale_factor_y = 2.0
    # scale_factor_x = 2.0 * jnp.pi
    # scale_factor_y = 2.0 * jnp.pi
    domain = Domain(
        (Nx, Ny), (True, True), scale_factors=(scale_factor_x, scale_factor_y)
    )
    # domain = Domain((Nx, Ny), (True, True))

    u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x) * jnp.cos(
        X[1] * 2 * jnp.pi / scale_factor_y
    )
    u = Field.FromFunc(domain, func=u_fn, name="u_2d")

    u_x_fn = (
        lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * 2
        * jnp.pi
        / scale_factor_x
    )
    u_x_ana = Field.FromFunc(domain, func=u_x_fn, name="u_2d_x_ana")
    u_xx_fn = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_x) ** 2
    )
    u_xx_ana = Field.FromFunc(domain, func=u_xx_fn, name="u_2d_xx_ana")

    u_y_fn = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin(X[1] * 2 * jnp.pi / scale_factor_y)
        * 2
        * jnp.pi
        / scale_factor_y
    )
    u_y_ana = Field.FromFunc(domain, func=u_y_fn, name="u_2d_y_ana")
    u_yy_fn = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_y) ** 2
    )
    u_yy_ana = Field.FromFunc(domain, func=u_yy_fn, name="u_2d_yy_ana")

    u_int_x_fn = (
        lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_x) ** (-1)
    )
    u_int_x_ana = Field.FromFunc(domain, func=u_int_x_fn, name="u_2d_int_x_ana")
    u_int_xx_fn = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_x) ** (-2)
    )
    u_int_xx_ana = Field.FromFunc(domain, func=u_int_xx_fn, name="u_2d_int_xx_ana")

    u_int_y_fn = (
        lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_y) ** (-1)
    )
    u_int_y_ana = Field.FromFunc(domain, func=u_int_y_fn, name="u_2d_int_y_ana")
    u_int_yy_fn = (
        lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
        * (2 * jnp.pi / scale_factor_y) ** (-2)
    )
    u_int_yy_ana = Field.FromFunc(domain, func=u_int_yy_fn, name="u_2d_int_yy_ana")

    u_hat = u.hat()

    u_hat_int_x = u_hat.integrate(0, 1)
    u_hat_int_xx = u_hat.integrate(0, 2)
    u_hat_int_y = u_hat.integrate(1, 1)
    u_hat_int_yy = u_hat.integrate(1, 2)

    u_int_x = u_hat_int_x.no_hat()
    u_int_xx = u_hat_int_xx.no_hat()
    u_int_y = u_hat_int_y.no_hat()
    u_int_yy = u_hat_int_yy.no_hat()

    u_hat_x = u_hat.diff(0, 1)
    u_hat_x_2 = u_hat.diff(0, 2)
    u_hat_y = u_hat.diff(1, 1)
    u_hat_y_2 = u_hat.diff(1, 2)
    u_x = u_hat_x.no_hat()
    u_x.name = "u_2d_x"
    u_x_2 = u_hat_x_2.no_hat()
    u_x_2.name = "u_2d_x_2"
    u_y = u_hat_y.no_hat()
    u_y.name = "u_2d_y"
    u_y_2 = u_hat_y_2.no_hat()
    u_y_2.name = "u_2d_y_2"
    u_int_x.plot(u_int_x_ana)
    # u_x.plot()
    # u_y.plot()
    # u.plot_center(0, u_x, u_x_2, u_int_x, u_int_xx)
    # u.plot_center(1, u_y, u_y_2, u_int_y, u_int_yy)
    tol = 1e-5
    # print(abs(u_x - u_x_ana))
    # print(abs(u_x_2 - u_xx_ana))
    # print(abs(u_y - u_y_ana))
    # print(abs(u_y_2 - u_yy_ana))
    # print(abs(u_int_x - u_int_x_ana))
    # print(abs(u_int_xx - u_int_xx_ana))
    # print(abs(u_int_y - u_int_y_ana))
    # print(abs(u_int_yy - u_int_yy_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_x_2 - u_xx_ana) < tol
    assert abs(u_y - u_y_ana) < tol
    assert abs(u_y_2 - u_yy_ana) < tol
    assert abs(u_int_x - u_int_x_ana) < tol
    assert abs(u_int_xx - u_int_xx_ana) < tol
    assert abs(u_int_y - u_int_y_ana) < tol
    assert abs(u_int_yy - u_int_yy_ana) < tol


def test_fourier_simple_3D():
    Nx = 24
    Ny = Nx + 4
    Nz = 20
    scale_factor_x = 1.0
    scale_factor_z = 2.0
    domain = Domain(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(scale_factor_x, 1.0, scale_factor_z),
    )

    # u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi/2)
    u_0_fn = (
        lambda X: 0.0
        * jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    u_fn = (
        lambda X: 1.0 *
        (jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x))
        + jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x))
        + jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
        + jnp.sin(X[1] * jnp.pi)
        + 1.0
        * jnp.cos(X[0] * 4 * jnp.pi / scale_factor_x)
        * jnp.cos(X[2] * 4 * jnp.pi / scale_factor_z)
        + 1.0
        * jnp.cos(X[0] * jnp.pi * 6 / scale_factor_x)
        * jnp.cos(X[2] * jnp.pi * 6 / scale_factor_z)
    )
    u = Field.FromFunc(domain, func=u_fn, name="u_3d")
    v = Field.FromFunc(domain, func=u_0_fn, name="v_3d")
    w = Field.FromFunc(domain, func=u_0_fn, name="w_3d")
    U = VectorField([u, v, w])
    U_hat = U.hat()
    u_hat = U_hat[0]
    # v_hat = U_hat[1]
    # w_hat = U_hat[2]
    u_x = u.diff(0, 1)
    u_xx = u.diff(0, 2)
    u_y = u.diff(1, 1)
    u_yy = u.diff(1, 2)
    u_z = u.diff(2, 1)
    u_zz = u.diff(2, 2)
    # u_int_x = u.integrate(0, 1)
    # u_int_xx = u.integrate(0, 2)
    # u_int_yy = u.integrate(1, 2)
    # u_int_z = u.integrate(2, 1)
    # u_int_zz = u.integrate(2, 2)

    u_hat_x = u_hat.diff(0, 1)
    u_hat_xx = u_hat.diff(0, 2)
    u_hat_y = u_hat.diff(1, 1)
    u_hat_yy = u_hat.diff(1, 2)
    u_hat_z = u_hat.diff(2, 1)
    u_hat_zz = u_hat.diff(2, 2)
    # u_int_hat_x = u_hat.integrate(0, 1)
    # u_int_hat_xx = u_hat.integrate(0, 2)
    # u_int_hat_yy = u_hat.integrate(1, 2)
    # u_int_hat_z = u_hat.integrate(2, 1)
    # u_int_hat_zz = u_hat.integrate(2, 2)

    # U_nohat.plot(U)
    # u.plot()
    u_x.plot(u_hat_x.no_hat())
    u_xx.plot(u_hat_xx.no_hat())
    # u_int_x.plot(u)
    # u_int_hat_x.no_hat().plot(u)
    # u_xx.plot(u_hat_xx.no_hat())
    # print(abs(U - U_nohat))
    # print(abs(u_hat_x.no_hat() - u_x))
    # print(abs(u_hat_xx.no_hat() - u_xx))
    # print(abs(u_hat_y.no_hat() - u_y))
    # print(abs(u_hat_yy.no_hat() - u_yy))
    # print(abs(u_hat_z.no_hat() - u_z))
    # print(abs(u_hat_zz.no_hat() - u_zz))
    # print(abs(u_int_hat_x.no_hat() - u_int_x))
    # print(abs(u_int_hat_xx.no_hat() - u_int_xx))
    # print(abs(u_int_hat_yy.no_hat() - u_int_yy))
    # print(abs(u_int_hat_z.no_hat() - u_int_z))
    # print(abs(u_int_hat_zz.no_hat() - u_int_zz))
    tol = 1e-9
    # assert abs(U - U_nohat) < tol
    assert abs(u_hat_x.no_hat() - u_x) < tol
    assert abs(u_hat_xx.no_hat() - u_xx) < tol
    assert abs(u_hat_y.no_hat() - u_y) < tol
    assert abs(u_hat_yy.no_hat() - u_yy) < tol
    assert abs(u_hat_z.no_hat() - u_z) < tol
    assert abs(u_hat_zz.no_hat() - u_zz) < tol
    # assert abs(u_int_hat_x.no_hat() - u_int_x) < tol
    # assert abs(u_int_hat_xx.no_hat() - u_int_xx) < tol
    # assert abs(u_int_hat_yy.no_hat() - u_int_yy) < tol
    # assert abs(u_int_hat_z.no_hat() - u_int_z) < tol
    # assert abs(u_int_hat_zz.no_hat() - u_int_zz) < tol
    # TODO test integration


def test_cheb_integration_1D():
    Nx = 24
    domain = Domain((Nx,), (False,))

    u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u_1d")

    u_fn_int = lambda X: -((2 / jnp.pi) ** 2) * jnp.cos(X[0] * jnp.pi / 2)
    u_int_ana = Field.FromFunc(domain, func=u_fn_int, name="u_1d_int_ana")
    u_int = u.integrate(0, order=2, bc_left=0.0, bc_right=0.0)
    u_int.name = "u_int"
    u_int.plot(u, u_int_ana)

    tol = 1e-7
    # print(abs(u_int - u_int_ana))
    assert abs(u_int - u_int_ana) < tol


def test_cheb_integration_2D():
    Nx = 24
    Ny = Nx + 4
    domain = Domain((Nx, Ny), (True, False))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u_2d")

    u_fn_int_1 = lambda X: jnp.cos(X[0]) * (2 / jnp.pi) * jnp.sin(X[1] * jnp.pi / 2) + (
        2 / jnp.pi
    ) * jnp.cos(X[0])
    u_int_1_ana = Field.FromFunc(domain, func=u_fn_int_1, name="u_2d_int_1_ana")
    u_fn_int_2 = (
        lambda X: -jnp.cos(X[0]) * (2 / jnp.pi) ** 2 * jnp.cos(X[1] * jnp.pi / 2)
    )
    u_int_2_ana = Field.FromFunc(domain, func=u_fn_int_2, name="u_2d_int_2_ana")
    u_int_1 = u.integrate(1, order=1, bc_left=0.0)
    u_int_2 = u.integrate(1, order=2, bc_left=0.0, bc_right=0.0)
    u_int_1.name = "u_int_1"
    u_int_2.name = "u_int_2"
    # u_int_1.plot(u, u_int_1_ana)
    # u_int_2.plot(u, u_int_2_ana)

    tol = 1e-7
    # print(abs(u_int_1 - u_int_1_ana))
    # print(abs(u_int_2 - u_int_2_ana))
    assert abs(u_int_1 - u_int_1_ana) < tol
    assert abs(u_int_2 - u_int_2_ana) < tol


def test_cheb_integration_3D():
    Nx = 24
    Ny = Nx + 4
    Nz = Nx - 4
    domain = Domain((Nx, Ny, Nz), (True, False, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u = Field.FromFunc(domain, func=u_fn, name="u_3d")

    u_fn_int = (
        lambda X: -jnp.cos(X[0])
        * jnp.cos(X[2])
        * (2 / jnp.pi) ** 2
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    u_int_ana = Field.FromFunc(domain, func=u_fn_int, name="u_3d_int_ana")
    u_int = u.integrate(1, order=2, bc_left=0.0, bc_right=0.0)
    u_int.name = "u_int"
    # u_int.plot(u_int_ana)

    tol = 1e-8
    # print(abs(u_int - u_int_ana))
    assert abs(u_int - u_int_ana) < tol

def test_definite_integral():
    tol = 1e-10
    # 1D
    # Fourier
    Nx = 60
    sc_x = 1.0
    domain_1D_fourier = Domain((Nx,), (True,), scale_factors=(sc_x,))
    u_fn_1d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x))
    u_1d_fourier = Field.FromFunc(domain_1D_fourier, u_fn_1d_fourier, name="u_1d_fourier")
    assert abs(u_1d_fourier.definite_integral(0) - 1.2660658777520084) < tol
    # Chebyshev
    domain_1D_cheb = Domain((Nx,), (False,))
    u_fn_1d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - 1
    u_1d_cheb = Field.FromFunc(domain_1D_cheb, u_fn_1d_cheb, name="u_1d_cheb")
    # print(u_1d_cheb.definite_integral(0))
    # print(abs(u_1d_cheb.definite_integral(0) - 0.5321317555))
    assert abs(u_1d_cheb.definite_integral(0) - 0.5321317555) < tol
    # 2D
    # Fourier
    Ny = 64
    sc_y = 2.0
    domain_2D_fourier = Domain((Nx,Ny), (True,True), scale_factors=(sc_x,sc_y))
    u_fn_2d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y)**2)
    u_2d_fourier = Field.FromFunc(domain_2D_fourier, u_fn_2d_fourier, name="u_2d_fourier")
    # print(abs(u_2d_fourier.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
    assert (abs(u_2d_fourier.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol
    assert (abs(u_2d_fourier.volume_integral() - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol
    # Chebyshev
    domain_2D_cheb = Domain((Nx,Ny), (False,False))
    u_fn_2d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y)**2)
    u_2d_cheb = Field.FromFunc(domain_2D_cheb, u_fn_2d_cheb, name="u_2d_cheb")
    # print(abs(u_2d_cheb.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
    # print((u_2d_cheb.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
    assert (abs(u_2d_cheb.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol
    assert (abs(u_2d_cheb.volume_integral() - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol
    # Mixed
    domain_2D_mixed = Domain((Nx,Ny), (False,True), scale_factors=(1.0, sc_y))
    u_2d_mixed = Field.FromFunc(domain_2D_mixed, u_fn_2d_cheb, name="u_2d_mixed")
    # print(abs(u_2d_mixed.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
    assert (abs(u_2d_mixed.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol
    assert (abs(u_2d_mixed.volume_integral() - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol
    domain_2D_mixed_2 = Domain((Nx,Ny), (True,False), scale_factors=(sc_x, 1.0))
    u_2d_mixed_2 = Field.FromFunc(domain_2D_mixed_2, u_fn_2d_cheb, name="u_2d_mixed_2")
    # print(abs(u_2d_mixed_2.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
    assert (abs(u_2d_mixed_2.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol
    assert (abs(u_2d_mixed_2.volume_integral() - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol
    # 3D
    # Fourier
    # Nx = 96
    # Ny = 96
    Nz = 96
    sc_z = 3.0
    domain_3D_fourier = Domain((Nx,Ny,Nz), (True,True,True), scale_factors=(sc_x,sc_y,sc_z))
    u_fn_3d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y)**2) * jnp.exp(jnp.cos(X[2] * 2 * jnp.pi / sc_z)**2)
    u_3d_fourier = Field.FromFunc(domain_3D_fourier, u_fn_3d_fourier, name="u_3d_fourier")
    # print(u_3d_fourier.definite_integral(2).definite_integral(1).definite_integral(0) - -10.84981433261992)
    assert (abs(u_3d_fourier.definite_integral(2).definite_integral(1).definite_integral(0) - -10.84981433261992)) < tol
    assert (abs(u_3d_fourier.volume_integral()- -10.84981433261992)) < tol
    # Chebyshev
    domain_3D_cheb = Domain((Nx,Ny,Nz), (False,False,False))
    u_fn_3d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi)**2) + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi)**2)
    u_3d_cheb = Field.FromFunc(domain_3D_cheb, u_fn_3d_cheb, name="u_3d_cheb")
    # print(u_3d_cheb.definite_integral(2).definite_integral(1).definite_integral(0) - 10.128527022082872)
    assert (abs(u_3d_cheb.definite_integral(2).definite_integral(1).definite_integral(0) - 10.128527022082872)) < tol
    assert (abs(u_3d_cheb.volume_integral()- 10.128527022082872)) < tol
    # Mixed
    domain_3D_mixed = Domain((Nx,Ny,Nz), (True,False,True), scale_factors=(sc_x, 1.0, sc_z))
    u_fn_3d_mixed = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi)**2) + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi / sc_z)**2)
    u_3d_mixed = Field.FromFunc(domain_3D_mixed, u_fn_3d_mixed, name="u_3d_mixed")
    # print(u_3d_mixed.definite_integral(2).definite_integral(1).definite_integral(0) - 7.596395266449558)
    assert (abs(u_3d_mixed.definite_integral(2).definite_integral(1).definite_integral(0) - 7.596395266449558)) < tol
    assert (abs(u_3d_mixed.volume_integral()- 7.596395266449558)) < tol

def test_poisson_slices():
    Nx = 24
    Ny = Nx + 4
    Nz = Nx - 4
    # Nx = 4
    # Ny = 4
    # Nz = 6
    scale_factor_x = 1.0
    scale_factor_z = 1.0

    domain = Domain(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(scale_factor_x, 1.0, scale_factor_z),
    )
    domain_y = Domain((Ny,), (False))

    rhs_fn = (
        lambda X: -(
            (2 * jnp.pi / scale_factor_x) ** 2
            + jnp.pi**2 / 4
            + (2 * jnp.pi / scale_factor_z) ** 2
        )
        * jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    rhs = Field.FromFunc(domain, rhs_fn, name="rhs")

    u_ana_fn = (
        lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    u_ana = Field.FromFunc(domain, u_ana_fn, name="u_ana")
    rhs_hat = rhs.hat()
    rhs_nohat = rhs_hat.no_hat()

    mat = rhs_hat.assemble_poisson_matrix()

    def solve_poisson_for_single_wavenumber(kx, kz):
        # kx, kz = int(kx_), int(kz_)
        if kx == 0 or kz == 0:
            # assumes homogeneneous Dirichlet boundary conditions
            return rhs_hat[kx, :, kz] * 0.0
            # return FourierFieldSlice(
            #     domain_y, 1, rhs_hat[kx, :, kz] * 0.0, "rhs_t_slice", kx, kz
            # )
        rhs_hat_slice = FourierFieldSlice(
            domain_y,
            1,
            rhs_hat[kx, :, kz],
            "rhs_hat_slice",
            rhs_hat.domain.grid[0][kx],
            rhs_hat.domain.grid[2][kz],
            ks_int=[kx, kz],
        )
        out = rhs_hat_slice.solve_poisson(mat)
        # out = rhs_hat_slice.solve_poisson()
        return out.field

    start_time = time.time()
    out_hat = rhs_hat.reconstruct_from_wavenumbers(solve_poisson_for_single_wavenumber)
    print(str(time.time() - start_time) + " seconds used for reconstruction.")
    out = out_hat.no_hat()

    # u_ana.plot(out)

    tol = 1e-8
    # print(abs(u_ana - out))
    assert abs(u_ana - out) < tol


def test_poisson_no_slices():
    Nx = 20
    Ny = 28
    Nz = 24

    # scale_factor_x = 1.0
    # scale_factor_z = 1.5
    scale_factor_x = 2 * jnp.pi
    scale_factor_z = 2 * jnp.pi
    domain = Domain(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(scale_factor_x, 1.0, scale_factor_z),
    )

    rhs_fn = (
        lambda X: -(
            (2 * jnp.pi / scale_factor_x) ** 2
            + jnp.pi**2 / 4
            + (2 * jnp.pi / scale_factor_z) ** 2
        )
        * jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    rhs = Field.FromFunc(domain, rhs_fn, name="rhs")

    u_ana_fn = (
        lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
        * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
        * jnp.cos(X[1] * jnp.pi / 2)
    )

    u_ana = Field.FromFunc(domain, u_ana_fn, name="u_ana")
    rhs_hat = rhs.hat()

    out_hat = rhs_hat.solve_poisson()
    out = out_hat.no_hat()

    # u_ana.plot(out)

    tol = 1e-8
    # print(abs(u_ana - out))
    assert abs(u_ana - out) < tol


def test_navier_stokes_laminar(Ny=96, pertubation_factor=0.1):
    Re = 1.5e0

    end_time = 8
    NavierStokesVelVort.max_dt = 1e10
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=24,
        Ny=Ny,
        Nz=16,
        end_time=end_time,
        pertubation_factor=pertubation_factor,
    )
    nse.solve()

    vel_x_fn_ana = lambda X: -1 * nse.u_max_over_u_tau * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana")

    vel_0 = nse.get_initial_field("velocity_hat").no_hat()
    print("Doing post-processing")
    for i in jnp.arange(nse.time_step)[-4:]:
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        vel[0].plot_center(1, vel_0[0], vel_x_ana)
        vel[1].plot_center(1, vel_0[1])
        vel[2].plot_center(1, vel_0[2])
        tol = 6e-5
        print(abs(vel[0] - vel_x_ana))
        print(abs(vel[1]))
        print(abs(vel[2]))
        print("max vel: " + str(vel[0].max() / vel_0[0].max()))
        # check that the simulation is really converged
        assert abs(vel[0] - vel_x_ana) < tol
        assert abs(vel[1]) < tol
        assert abs(vel[2]) < tol


def test_navier_stokes_laminar_convergence():
    Nys = [24, 48, 96]
    end_time = 10

    def run(Ny):
        nse = solve_navier_stokes_laminar(
            Re=1, end_time=end_time, Ny=Ny, pertubation_factor=0
        )
        nse.solve()
        vel_x_fn_ana = (
            lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
        )
        # vel_x_ana = Field.FromFunc(nse.domain, vel_x_fn_ana, name="vel_x_ana")
        vel_hat = nse.get_latest_field("velocity_hat")
        vel = vel_hat.no_hat()
        return vel[0].l2error(vel_x_fn_ana)

    errors = list(map(run, Nys))
    errorsLog = list(map(lambda x: jnp.log2(x), errors))
    print(errors)

    def fittingFunc(x, a, b):
        return a + b * x

    result = optimization.curve_fit(fittingFunc, Nys, errorsLog)
    print(result)


def test_optimization():
    Re = 1e0
    Ny = 24
    end_time = 1

    nse = solve_navier_stokes_laminar(
        Re=Re, Ny=Ny, end_time=end_time, pertubation_factor=0.0
    )

    def run(v0):
        nse_ = solve_navier_stokes_laminar(
            Re=Re, Ny=Ny, end_time=end_time, max_iter=10, pertubation_factor=0.0
        )
        nse_.max_iter = 10
        v0_field = Field(nse_.domain_no_hat, v0)
        vel_0 = nse_.get_initial_field("velocity_hat")
        vel_0_new = VectorField([v0_field.hat(), vel_0[1], vel_0[2]])
        nse_.set_field("velocity_hat", 0, vel_0_new)
        nse_.after_time_step_fn = None
        nse_.solve()
        vel_out = nse_.get_latest_field("velocity_hat").no_hat()
        return (vel_out[0].max() / vel_0[0].max()).real

    vel_x_fn_ana = (
        lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    )
    v0_0 = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana)

    v0s = [v0_0.field]
    eps = 1e3
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_field = Field(nse.domain_no_hat, corr, name="correction")
        corr_field.update_boundary_conditions()
        print("gain: " + str(gain))
        print("corr (abs): " + str(abs(corr_field)))
        v0s.append(v0s[-1] + eps * corr_field.field)
        v0_new = Field(nse.domain_no_hat, v0s[-1])
        v0_new.name = "vel_0_" + str(i)
        v0_new.plot(v0_0)


def test_navier_stokes_turbulent():
    Re = 1.8e6

    end_time = 50
    s_x = 1.87
    s_z = 0.93
    # s_x = 2 * jnp.pi
    # s_z = 2 * jnp.pi
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Ny=60,
        Nx=64,
        end_time=end_time,
        pertubation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, pertubation_factor=1
        # Re=Re, Ny=48, Nx=24, end_time=end_time, pertubation_factor=1
    )

    def vortex_fun(center_y, center_z, a):
        def ret(X):
            _, y, z = X[0], X[1], X[2]
            u = 0
            v = -a * (z - center_z) / s_z
            w = a * (y - center_y) / 2
            return (u, v, w)

        return ret

    def ts_vortex_fun(center_x, center_z, a):
        def ret(X):
            # x,_,z = X
            x, _, z = X[0], X[1], X[2]
            u = a * (z - center_z) / s_z
            v = 0
            w = -a * (x - center_x) / s_x
            return (u, v, w)

        return ret

    vortex_1_fun = vortex_fun(0.0, s_z / 2, 0.1)
    vortex_2_fun = vortex_fun(0.0, -s_z / 2, 0.1)
    ts_vortex_1_fun = ts_vortex_fun(s_x / 2, 0.0, 0.1)
    ts_vortex_2_fun = ts_vortex_fun(-s_x / 2, 0.0, 0.1)
    # vortex_sum = lambda X: vortex_1_fun(X) + vortex_2_fun(X)
    vortex_sum = lambda X: [
        ts_vortex_1_fun(X)[i]
        + ts_vortex_2_fun(X)[i]
        + vortex_1_fun(X)[i]
        + vortex_2_fun(X)[i]
        for i in range(3)
    ]

    # Add small velocity perturbations localized to the shear layers
    omega = 0.05

    Ly = 2.0
    Lz = s_z
    # vel_x_fn = lambda X: jnp.pi / 3 * jnp.cos(
    #     X[1] * jnp.pi / 2) + (1 - X[1]**2) * vortex_sum(X)[0]
    vel_x_fn = (
        lambda X: ((0.5 + X[1] / Ly) * (0.5 - X[1] / Ly)) / 0.25
        + 0.1 * jnp.sin(2 * jnp.pi * X[2] / Lz * omega)
        + 0 * X[0]
    )
    vel_x = Field.FromFunc(nse.domain_no_hat, vel_x_fn, name="velocity_x")

    # vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) * vortex_sum(X)[1]
    vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + 0.1 * jnp.sin(
        2 * jnp.pi * X[2] / Lz * omega
    )
    # vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) *vortex_sum(X)[2]
    vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2]
    vel_y = Field.FromFunc(nse.domain_no_hat, vel_y_fn, name="velocity_y")
    vel_z = Field.FromFunc(nse.domain_no_hat, vel_z_fn, name="velocity_z")
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 2

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel[0].plot_3d()
    vel[1].plot_3d()
    vel[2].plot_3d()

    def after_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_field("velocity_hat", i).no_hat()
            vort_hat, _ = nse.get_vorticity_and_helicity()
            vort = vort_hat.no_hat()
            vel[0].plot_3d()
            vel[1].plot_3d()
            vel[2].plot_3d()
            vort[0].plot_3d()
            vort[1].plot_3d()
            vort[2].plot_3d()
            vel[0].plot_center(1)
            vel[1].plot_center(1)
            vel[2].plot_center(1)

    nse.after_time_step_fn = after_time_step
    # nse.after_time_step_fn = None
    nse.solve()
    return nse.get_latest_field("velocity_hat").no_hat().field


def test_vmap():
    def fn(x):
        return jax.lax.cond(x > 3, lambda x_: x_ * 2, lambda x_: x_ * 3, x)
        # if x > 3:
        #     return x*2
        # else:
        #     return x*3

    def fn2(X):
        x = X[0]
        y = X[1]
        A = jnp.zeros((4, 4, 24, 24))
        B = jnp.ones((4, 24, 4))
        # print(A[x, :, y])
        # return A[x,:,  y] + B[x, :, y]
        # return jnp.dot(x,y)
        out = jax.lax.cond(x + y == 0, lambda: x * y, lambda: x * y / (x + y))
        return out

    N = 100
    xs = jnp.arange(N, dtype=float)
    ys = jnp.arange(N, dtype=float)
    Xs, Ys = jnp.meshgrid(xs, ys)
    X = jnp.array(list(zip(Xs.flatten(), Ys.flatten())))
    # out = jax.vmap(fn)(xs)
    # print(xs)
    # print(out)
    # fn2vmap = jnp.vectorize(fn2, signature='(n),(m)->(k)')
    # fn2vmap = jnp.vectorize(fn2)
    fn2_jit = jax.jit(fn2)
    start = time.time()
    # for x in X:
    #     fn2(x)
    [fn2_jit(x) for x in X]
    end = time.time()
    # for x in xs:
    #     for y in ys:
    #         print(fn2(x, y))
    fn2vmap = jax.vmap(fn2_jit)
    start_2 = time.time()
    fn2vmap(X).reshape(N, N)
    end_2 = time.time()
    print("Elapsed time non-vectorized version: ", end - start)
    print("Elapsed time vectorized version: ", end_2 - start_2)
    # print(fn2vmap(xs, ys))


def test_linear_stability():
    n = 64
    Re = 5772.22
    alpha = 1.02056

    lsc = LinearStabilityCalculation(Re, alpha, n)
    evs, _ = lsc.calculate_eigenvalues()
    # print(evs[0])
    # print(evecs[0])
    assert evs[0].real <= 0.0 and evs[0].real >= -1e-8


def test_pseudo_2d():
    Ny = 64
    # Ny = 24
    # Re = 5772.22
    # Re = 6000
    Re = 5500
    alpha = 1.02056
    # alpha = 1.0

    Nx = 496
    Nz = 4
    lsc = LinearStabilityCalculation(Re, alpha, Ny)

    end_time = 100
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(4 * (2 * jnp.pi / alpha), 1.0, 1.0),
    )

    make_field_file_name = (
        lambda field_name: field_name
        + "_"
        + str(Re)
        + "_"
        + str(Nx)
        + "_"
        + str(Ny)
        + "_"
        + str(Nz)
    )
    try:
        # raise FileNotFoundError()
        u = Field.FromFile(nse.domain_no_hat, make_field_file_name("u"), name="u_pert")
        v = Field.FromFile(nse.domain_no_hat, make_field_file_name("v"), name="v_pert")
        w = Field.FromFile(nse.domain_no_hat, make_field_file_name("w"), name="w_pert")
        print("found existing fields, skipping eigenvalue computation")
        lsc.velocity_field_ = (u, v, w)
    except FileNotFoundError:
        print("could not find fields")
        u, v, w = lsc.velocity_field(nse.domain_no_hat)
    u.save_to_file(make_field_file_name("u"))
    v.save_to_file(make_field_file_name("v"))
    w.save_to_file(make_field_file_name("w"))
    vel_x_hat, _, _ = nse.get_initial_field("velocity_hat")

    eps = 5e-3
    nse.init_velocity(
        VectorField(
            [
                vel_x_hat + eps * u.hat(),
                eps * v.hat(),
                eps * w.hat(),
            ]
        ),
    )

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.domain_no_hat)
    energy_over_time_fn = lambda t: eps**2 * energy_over_time_fn_raw(t)
    energy_x_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(nse.domain_no_hat)[0](t, 0)
    energy_y_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(nse.domain_no_hat)[0](t, 1)
    print("eigenvalue: ", ev)
    plot_interval = 10

    vel_pert_0 = nse.get_initial_field("velocity_hat").no_hat()[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    energy_x_t = []
    energy_y_t = []
    energy_t_ana = []
    energy_x_t_ana = []
    energy_y_t_ana = []
    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            vel_x_max = vel[0].max()
            print("vel_x_max: ", vel_x_max)
            vel_x_fn_ana = lambda X: - vel_x_max * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            vel_x_ana = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana")
            # vel_1_lap_a = nse.get_field("v_1_lap_hat_a", i).no_hat()
            # vel_1_lap_a.plot_3d()
            vel_pert = VectorField([vel[0] - vel_x_ana, vel[1], vel[2]])
            vel_hat_old = nse.get_field("velocity_hat", max(0, i-1))
            vel_old = vel_hat_old.no_hat()
            vel_x_max_old = vel_old[0].max()
            vel_x_fn_ana_old = lambda X: - vel_x_max_old * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            vel_x_ana_old = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana_old, name="vel_x_ana_old")
            vel_pert_old = VectorField([vel_old[0] - vel_x_ana_old, vel_old[1], vel_old[2]])
            vel_pert_energy = 0
            v_1_lap_p = nse.get_latest_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p_0 = nse.get_initial_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p.time_step = i
            v_1_lap_p.plot_3d(2)
            v_1_lap_p.plot_center(0)
            v_1_lap_p.plot_center(1, v_1_lap_p_0)
            for j in range(2):
                vel[j].time_step = i
                vel_pert[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vel[j].plot_3d(2)
                # vel[j].plot_center(0)
                if j == 0:
                    vel[j].plot_center(1, vel_x_ana)
                elif j == 1:
                    vel[j].plot_center(1, vel_pert_0)
                # vel_hat[j].plot_3d()
                vel_pert[j].name = "velocity_pertubation_" + "xyz"[j]
                vel_pert[j].plot_3d()
                vel_pert[j].plot_3d(2)
                # vel_pert[j].plot_center(0)
                # vel_pert[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            print("analytical velocity pertubation energy: ", energy_over_time_fn(nse.time))
            print("velocity pertubation energy: ", vel_pert_energy)
            print("velocity pertubation energy x: ", vel_pert[0].energy())
            print("analytical velocity pertubation energy x: ", energy_x_over_time_fn(nse.time))
            print("velocity pertubation energy y: ", vel_pert[1].energy())
            print("analytical velocity pertubation energy y: ", energy_y_over_time_fn(nse.time))
            print("velocity pertubation energy z: ", vel_pert[1].energy())
            vel_pert_energy_old = vel_pert_old.energy()
            if vel_pert_energy - vel_pert_energy_old >= 0:
                print("velocity pertubation energy increase: ", vel_pert_energy - vel_pert_energy_old)
            else:
                print("velocity pertubation energy decrease: ", - (vel_pert_energy - vel_pert_energy_old))
            print("velocity pertubation energy x change: ", vel_pert[0].energy() - vel_pert_old[0].energy())
            print("velocity pertubation energy y change: ", vel_pert[1].energy() - vel_pert_old[1].energy())
            print("velocity pertubation energy z change: ", vel_pert[2].energy() - vel_pert_old[2].energy())
            ts.append(nse.time)
            energy_t.append(vel_pert_energy)
            energy_x_t.append(vel_pert[0].energy())
            energy_y_t.append(vel_pert[1].energy())
            energy_t_ana.append(energy_over_time_fn(nse.time))
            energy_x_t_ana.append(energy_x_over_time_fn(nse.time))
            energy_y_t_ana.append(energy_y_over_time_fn(nse.time))
            # if i > plot_interval * 3:
            if True:
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_t_ana)
                ax.plot(ts, energy_t, ".")
                fig.savefig("plots/energy_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_x_t_ana)
                ax.plot(ts, energy_x_t, ".")
                fig.savefig("plots/energy_x_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_y_t_ana)
                ax.plot(ts, energy_y_t, ".")
                fig.savefig("plots/energy_y_t.pdf")
        # input("carry on?")

    nse.after_time_step_fn = None
    nse.before_time_step_fn = before_time_step
    # nse.before_time_step_fn = None

    nse.solve()


def test_dummy_velocity_field():
    Re = 1e5

    end_time = 50

    nse = solve_navier_stokes_laminar(
        # Re=Re,
        # Ny=90,
        # Nx=64,
        # end_time=end_time,
        # pertubation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, pertubation_factor=1
        # Re=Re, Ny=60, Nx=32, end_time=end_time, pertubation_factor=1
        Re=Re,
        Ny=96,
        Nx=64,
        end_time=end_time,
        pertubation_factor=0,
    )

    sc_x = 1.87
    nse.max_iter = 1e10
    vel_x_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + jnp.cos(X[0] * 2*jnp.pi / sc_x) + jnp.cos(X[1] * 2*jnp.pi / 1.0)
    vel_y_fn = (
        # lambda X: 0.0 * X[0] * X[1] * X[2] + X[0] * X[2] * (1 - X[1] ** 2) ** 2
        lambda X: 0.0 * X[0] * X[1] * X[2] + jnp.cos(X[1] * 2*jnp.pi / 1.0)
    )  # fulfills bcs but breaks conti
    vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2]
    vel_x = Field.FromFunc(nse.domain_no_hat, vel_x_fn, name="velocity_x")
    vel_y = Field.FromFunc(nse.domain_no_hat, vel_y_fn, name="velocity_y")
    vel_z = Field.FromFunc(nse.domain_no_hat, vel_z_fn, name="velocity_z")
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 1

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana")

    def after_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_field("velocity_hat", i).no_hat()
            vort_hat, _ = nse.get_vorticity_and_helicity()
            vort = vort_hat.no_hat()
            vel_pert = VectorField([vel[0] - vel_x_ana, vel[1], vel[2]])
            vel[0].plot_3d()
            vel[1].plot_3d()
            vel[2].plot_3d()
            vort[0].plot_3d()
            vort[1].plot_3d()
            vort[2].plot_3d()
            vel[0].plot_center(0)
            vel[1].plot_center(0)
            vel[2].plot_center(0)
            vel[0].plot_center(1)
            vel[1].plot_center(1)
            vel[2].plot_center(1)
            vel_pert_energy = 0
            vel_pert_abs = 0
            for j in range(3):
                vel_pert_energy += vel_pert[j].energy()
                vel_pert_abs += abs(vel_pert[j])
            print("velocity pertubation energy: ", vel_pert_energy)
            print("velocity pertubation abs: ", vel_pert_abs)

    nse.after_time_step_fn = after_time_step
    # nse.after_time_step_fn = None
    nse.solve()
    return nse.get_latest_field("velocity_hat").no_hat().field


def test_pertubation_laminar(Ny=48, pertubation_factor=0.1):
    Re = 1.5e0

    end_time = 8
    nse = solve_navier_stokes_pertubation(
        Re=Re,
        Nx=16,
        Ny=Ny,
        Nz=16,
        end_time=end_time,
        pertubation_factor=pertubation_factor,
    )

    plot_interval = 1
    def before_time_step(nse):
        i = nse.time_step
        if (i) % plot_interval == 0:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            vel_pert = VectorField([vel[0], vel[1], vel[2]])
            vel_pert_energy = 0
            vort = vel.curl()
            for j in range(3):
                vel[j].time_step = i
                vort[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vort[j].name = "vorticity_" + "xyz"[j]
                # vel[j].plot_3d()
                vel[j].plot_3d(2)
                vort[j].plot_3d(2)
                vel[j].plot_center(0)
                vel[j].plot_center(1)
                vel_pert_energy += vel_pert[j].energy()
            print("velocity pertubation: ", vel_pert_energy)
            print("velocity pertubation x: ", vel_pert[0].energy())
            print("velocity pertubation y: ", vel_pert[1].energy())
            print("velocity pertubation z: ", vel_pert[2].energy())
        # input("carry on?")

    nse.before_time_step_fn = before_time_step
    # nse.max_dt = 1e10
    nse.solve()


    vel_0 = nse.get_initial_field("velocity_hat").no_hat()
    print("Doing post-processing")
    for i in jnp.arange(nse.time_step)[-4:]:
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        vel[0].plot_center(1, vel_0[0])
        vel[1].plot_center(1, vel_0[1])
        vel[2].plot_center(1, vel_0[2])
        tol = 1.7e-5
        print(abs(vel[0]))
        print(abs(vel[1]))
        print(abs(vel[2]))
        # check that the simulation is really converged
        assert abs(vel[0]) < tol
        assert abs(vel[1]) < tol
        assert abs(vel[2]) < tol


def test_pseudo_2d_pertubation():
    Ny = 64
    # Ny = 24
    # Re = 5772.22
    Re = 3000
    # Re = 6000
    alpha = 1.02056
    # alpha = 1.0

    Nx = 496
    Nz = 4
    lsc = LinearStabilityCalculation(Re, alpha, Ny)

    end_time = 10
    nse = solve_navier_stokes_pertubation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(4 * (2 * jnp.pi / alpha), 1.0, 1.0),
    )

    make_field_file_name = (
        lambda field_name: field_name
        + "_"
        + str(Re)
        + "_"
        + str(Nx)
        + "_"
        + str(Ny)
        + "_"
        + str(Nz)
    )
    try:
        # raise FileNotFoundError()
        u = Field.FromFile(nse.domain_no_hat, make_field_file_name("u"), name="u_pert")
        v = Field.FromFile(nse.domain_no_hat, make_field_file_name("v"), name="v_pert")
        w = Field.FromFile(nse.domain_no_hat, make_field_file_name("w"), name="w_pert")
        print("found existing fields, skipping eigenvalue computation")
    except FileNotFoundError:
        print("could not find fields")
        u, v, w = lsc.velocity_field(nse.domain_no_hat)
    u.save_to_file(make_field_file_name("u"))
    v.save_to_file(make_field_file_name("v"))
    w.save_to_file(make_field_file_name("w"))

    eps = 1e-3
    vel_x_hat, vel_y_hat, vel_z_hat = nse.get_initial_field("velocity_hat")
    nse.init_velocity(
        VectorField(
            [
                vel_x_hat + eps * u.hat(),
                eps * v.hat(),
                eps * w.hat(),
            ]
        ),
    )

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.domain_no_hat)
    energy_over_time_fn = lambda t: eps**2 * energy_over_time_fn_raw(t)
    energy_x_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(nse.domain_no_hat)[0](t, 0)
    energy_y_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(nse.domain_no_hat)[0](t, 1)
    print("eigenvalue: ", ev)
    plot_interval = 10

    vel_pert_0 = nse.get_initial_field("velocity_hat").no_hat()[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    energy_x_t = []
    energy_y_t = []
    energy_t_ana = []
    energy_x_t_ana = []
    energy_y_t_ana = []
    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            # vel_1_lap_a = nse.get_field("v_1_lap_hat_a", i).no_hat()
            # vel_1_lap_a.plot_3d()
            vel_pert = VectorField([vel[0], vel[1], vel[2]])
            vel_pert_old = nse.get_field("velocity_hat", max(0, i-1)).no_hat()
            vort = vel.curl()
            for j in range(3):
                vel[j].time_step = i
                vort[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vort[j].name = "vorticity_" + "xyz"[j]
                # vel[j].plot_3d()
                vel[j].plot_3d(2)
                vort[j].plot_3d(2)
                vel[j].plot_center(0)
                vel[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            print("analytical velocity pertubation energy: ", energy_over_time_fn(nse.time))
            print("velocity pertubation energy: ", vel_pert_energy)
            print("velocity pertubation energy x: ", vel_pert[0].energy())
            print("analytical velocity pertubation energy x: ", energy_x_over_time_fn(nse.time))
            print("velocity pertubation energy y: ", vel_pert[1].energy())
            print("analytical velocity pertubation energy y: ", energy_y_over_time_fn(nse.time))
            print("velocity pertubation energy z: ", vel_pert[2].energy())
            vel_pert_energy_old = vel_pert_old.energy()
            if vel_pert_energy - vel_pert_energy_old >= 0:
                print("velocity pertubation energy increase: ", vel_pert_energy - vel_pert_energy_old)
            else:
                print("velocity pertubation energy decrease: ", - (vel_pert_energy - vel_pert_energy_old))
            print("velocity pertubation energy x change: ", vel_pert[0].energy() - vel_pert_old[0].energy())
            print("velocity pertubation energy y change: ", vel_pert[1].energy() - vel_pert_old[1].energy())
            print("velocity pertubation energy z change: ", vel_pert[2].energy() - vel_pert_old[2].energy())
            ts.append(nse.time)
            energy_t.append(vel_pert_energy)
            energy_x_t.append(vel_pert[0].energy())
            energy_y_t.append(vel_pert[1].energy())
            energy_t_ana.append(energy_over_time_fn(nse.time))
            energy_x_t_ana.append(energy_x_over_time_fn(nse.time))
            energy_y_t_ana.append(energy_y_over_time_fn(nse.time))
            # if i > plot_interval * 3:
            if True:
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_t_ana)
                ax.plot(ts, energy_t, ".")
                fig.savefig("plots/energy_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_x_t_ana)
                ax.plot(ts, energy_x_t, ".")
                fig.savefig("plots/energy_x_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1,1)
                ax.plot(ts, energy_y_t_ana)
                ax.plot(ts, energy_y_t, ".")
                fig.savefig("plots/energy_y_t.pdf")
        # input("carry on?")

    nse.before_time_step_fn = before_time_step
    nse.after_time_step_fn = None

    nse.solve()


def run_all_tests():
    # test_1D_periodic()
    # test_1D_cheb()
    # test_2D()
    # test_3D()
    # test_fourier_1D()
    # test_fourier_2D()
    # test_fourier_simple_3D()
    # test_cheb_integration_1D()
    # test_cheb_integration_2D()
    # test_cheb_integration_3D()
    # test_definite_integral()
    # test_poisson_slices()
    # test_poisson_no_slices()
    # test_navier_stokes_laminar()
    # test_linear_stability()
    # test_navier_stokes_laminar_convergence()
    # test_optimization()
    # return test_navier_stokes_turbulent()
    # test_vmap()
    # test_transient_growth()
    # test_pseudo_2d()
    # test_dummy_velocity_field()
    # test_pertubation_laminar()
    test_pseudo_2d_pertubation()


def run_all_tests_profiling():
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # Run the operations to be profiled
    # run_all_tests()
    with Profile() as profile:
        run_all_tests()
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CALLS)
            .dump_stats("./navier-stokes.prof")
        )

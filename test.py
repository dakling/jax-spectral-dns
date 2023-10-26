#!/usr/bin/env python3

import unittest
import jax
import jax.numpy as jnp
from pathlib import Path
import os

# from cProfile import Profile
# from pstats import SortKey, Stats

# import numpy as np

from importlib import reload
import sys

jax.config.update("jax_enable_x64", True)

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

try:
    reload(sys.modules["examples"])
except:
    print("Unable to load examples")
from examples import run_pseudo_2d_pertubation

NoneType = type(None)


def init():
    newpaths = ['./fields/', "./plots/"]
    for newpath in newpaths:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    # clean plotting dir
    [f.unlink() for f in Path(newpaths[1]).glob("*.pdf") if f.is_file()]
    [f.unlink() for f in Path(newpaths[1]).glob("*.png") if f.is_file()]
    [f.unlink() for f in Path(newpaths[1]).glob("*.mp4") if f.is_file()]

class TestProject(unittest.TestCase):

    def setUp(self):
        init()

    def test_1D_cheb(self):
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
        # u.plot_center(0, u_x, u_xx)
        tol = 5e-4
        # print(abs(u_x - u_x_ana))
        # print(abs(u_xx - u_xx_ana))
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)


    def test_1D_periodic(self):
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

        # u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
        tol = 8e-5
        # print(abs(u_x - u_x_ana))
        # print(abs(u_xx - u_xx_ana))
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)


    def test_2D(self):
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

        # u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
        # u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
        tol = 5e-5
        # print(abs(u_x - u_x_ana))
        # print(abs(u_xx - u_xx_ana))
        # print(abs(u_y - u_y_ana))
        # print(abs(u_yy - u_yy_ana))
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_yy - u_yy_ana) < tol)


    def test_3D(self):
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

        # u.plot_center(0, u_x, u_xx, u_x_ana, u_xx_ana)
        # u.plot_center(1, u_y, u_yy, u_y_ana, u_yy_ana)
        # u.plot_center(2, u_z, u_zz, u_z_ana, u_zz_ana)

        # print(abs(u_x - u_x_ana))
        # print(abs(u_xx - u_xx_ana))
        # print(abs(u_y - u_y_ana))
        # print(abs(u_yy - u_yy_ana))
        # print(abs(u_z - u_z_ana))
        # print(abs(u_zz - u_zz_ana))
        tol = 5e-5
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_yy - u_yy_ana) < tol)
        self.assertTrue(abs(u_z - u_z_ana) < tol)
        self.assertTrue(abs(u_zz - u_zz_ana) < tol)


    def test_fourier_1D(self):
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
        # u_int.plot(u)
        # u.integrate(0).plot(u)
        # print(abs(u_int - u_int_ana))
        # print(abs(u_int_2 - u_int_ana_2))
        # print(abs(u_diff - u_diff_ana))
        # print(abs(u_diff_2 - u_diff_ana_2))
        self.assertTrue(abs(u_int - u_int_ana) < tol)
        self.assertTrue(abs(u_int_2 - u_int_ana_2) < tol)
        self.assertTrue(abs(u_diff - u_diff_ana) < tol)
        self.assertTrue(abs(u_diff_2 - u_diff_ana_2) < tol)


    def test_fourier_2D(self):
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
        # u_int_x.plot(u_int_x_ana)
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
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_x_2 - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_y_2 - u_yy_ana) < tol)
        self.assertTrue(abs(u_int_x - u_int_x_ana) < tol)
        self.assertTrue(abs(u_int_xx - u_int_xx_ana) < tol)
        self.assertTrue(abs(u_int_y - u_int_y_ana) < tol)
        self.assertTrue(abs(u_int_yy - u_int_yy_ana) < tol)


    def test_fourier_simple_3D(self):
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
        # u_x.plot(u_hat_x.no_hat())
        # u_xx.plot(u_hat_xx.no_hat())
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
        # self.assertTrue(abs(U - U_nohat) < tol)
        self.assertTrue(abs(u_hat_x.no_hat() - u_x) < tol)
        self.assertTrue(abs(u_hat_xx.no_hat() - u_xx) < tol)
        self.assertTrue(abs(u_hat_y.no_hat() - u_y) < tol)
        self.assertTrue(abs(u_hat_yy.no_hat() - u_yy) < tol)
        self.assertTrue(abs(u_hat_z.no_hat() - u_z) < tol)
        self.assertTrue(abs(u_hat_zz.no_hat() - u_zz) < tol)
        # self.assertTrue(abs(u_int_hat_x.no_hat() - u_int_x) < tol)
        # self.assertTrue(abs(u_int_hat_xx.no_hat() - u_int_xx) < tol)
        # self.assertTrue(abs(u_int_hat_yy.no_hat() - u_int_yy) < tol)
        # self.assertTrue(abs(u_int_hat_z.no_hat() - u_int_z) < tol)
        # self.assertTrue(abs(u_int_hat_zz.no_hat() - u_int_zz) < tol)
        # TODO test integration


    def test_cheb_integration_1D(self):
        Nx = 24
        domain = Domain((Nx,), (False,))

        u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
        u = Field.FromFunc(domain, func=u_fn, name="u_1d")

        u_fn_int = lambda X: -((2 / jnp.pi) ** 2) * jnp.cos(X[0] * jnp.pi / 2)
        u_int_ana = Field.FromFunc(domain, func=u_fn_int, name="u_1d_int_ana")
        u_int = u.integrate(0, order=2, bc_left=0.0, bc_right=0.0)
        u_int.name = "u_int"
        # u_int.plot(u, u_int_ana)

        tol = 1e-7
        # print(abs(u_int - u_int_ana))
        self.assertTrue(abs(u_int - u_int_ana) < tol)


    def test_cheb_integration_2D(self):
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
        self.assertTrue(abs(u_int_1 - u_int_1_ana) < tol)
        self.assertTrue(abs(u_int_2 - u_int_2_ana) < tol)


    def test_cheb_integration_3D(self):
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
        self.assertTrue(abs(u_int - u_int_ana) < tol)

    def test_definite_integral(self):
        tol = 1e-10
        # 1D
        # Fourier
        Nx = 60
        sc_x = 1.0
        domain_1D_fourier = Domain((Nx,), (True,), scale_factors=(sc_x,))
        u_fn_1d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x))
        u_1d_fourier = Field.FromFunc(domain_1D_fourier, u_fn_1d_fourier, name="u_1d_fourier")
        self.assertTrue(abs(u_1d_fourier.definite_integral(0) - 1.2660658777520084) < tol)
        # Chebyshev
        domain_1D_cheb = Domain((Nx,), (False,))
        u_fn_1d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - 1
        u_1d_cheb = Field.FromFunc(domain_1D_cheb, u_fn_1d_cheb, name="u_1d_cheb")
        # print(u_1d_cheb.definite_integral(0))
        # print(abs(u_1d_cheb.definite_integral(0) - 0.5321317555))
        self.assertTrue(abs(u_1d_cheb.definite_integral(0) - 0.5321317555) < tol)
        # 2D
        # Fourier
        Ny = 64
        sc_y = 2.0
        domain_2D_fourier = Domain((Nx,Ny), (True,True), scale_factors=(sc_x,sc_y))
        u_fn_2d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y)**2)
        u_2d_fourier = Field.FromFunc(domain_2D_fourier, u_fn_2d_fourier, name="u_2d_fourier")
        # print(abs(u_2d_fourier.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        self.assertTrue((abs(u_2d_fourier.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol)
        self.assertTrue((abs(u_2d_fourier.volume_integral() - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol)
        # Chebyshev
        domain_2D_cheb = Domain((Nx,Ny), (False,False))
        u_fn_2d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y)**2)
        u_2d_cheb = Field.FromFunc(domain_2D_cheb, u_fn_2d_cheb, name="u_2d_cheb")
        # print(abs(u_2d_cheb.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        # print((u_2d_cheb.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
        self.assertTrue((abs(u_2d_cheb.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol)
        self.assertTrue((abs(u_2d_cheb.volume_integral() - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol)
        # Mixed
        domain_2D_mixed = Domain((Nx,Ny), (False,True), scale_factors=(1.0, sc_y))
        u_2d_mixed = Field.FromFunc(domain_2D_mixed, u_fn_2d_cheb, name="u_2d_mixed")
        # print(abs(u_2d_mixed.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
        self.assertTrue((abs(u_2d_mixed.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol)
        self.assertTrue((abs(u_2d_mixed.volume_integral() - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962)) < tol)
        domain_2D_mixed_2 = Domain((Nx,Ny), (True,False), scale_factors=(sc_x, 1.0))
        u_2d_mixed_2 = Field.FromFunc(domain_2D_mixed_2, u_fn_2d_cheb, name="u_2d_mixed_2")
        # print(abs(u_2d_mixed_2.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        self.assertTrue((abs(u_2d_mixed_2.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol)
        self.assertTrue((abs(u_2d_mixed_2.volume_integral() - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812)) < tol)
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
        self.assertTrue((abs(u_3d_fourier.definite_integral(2).definite_integral(1).definite_integral(0) - -10.84981433261992)) < tol)
        self.assertTrue((abs(u_3d_fourier.volume_integral()- -10.84981433261992)) < tol)
        # Chebyshev
        domain_3D_cheb = Domain((Nx,Ny,Nz), (False,False,False))
        u_fn_3d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi)**2) + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi)**2)
        u_3d_cheb = Field.FromFunc(domain_3D_cheb, u_fn_3d_cheb, name="u_3d_cheb")
        # print(u_3d_cheb.definite_integral(2).definite_integral(1).definite_integral(0) - 10.128527022082872)
        self.assertTrue((abs(u_3d_cheb.definite_integral(2).definite_integral(1).definite_integral(0) - 10.128527022082872)) < tol)
        self.assertTrue((abs(u_3d_cheb.volume_integral()- 10.128527022082872)) < tol)
        # Mixed
        domain_3D_mixed = Domain((Nx,Ny,Nz), (True,False,True), scale_factors=(sc_x, 1.0, sc_z))
        u_fn_3d_mixed = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi)**2) + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi / sc_z)**2)
        u_3d_mixed = Field.FromFunc(domain_3D_mixed, u_fn_3d_mixed, name="u_3d_mixed")
        # print(u_3d_mixed.definite_integral(2).definite_integral(1).definite_integral(0) - 7.596395266449558)
        self.assertTrue((abs(u_3d_mixed.definite_integral(2).definite_integral(1).definite_integral(0) - 7.596395266449558)) < tol)
        self.assertTrue((abs(u_3d_mixed.volume_integral()- 7.596395266449558)) < tol)

    def test_poisson_slices(self):
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
        # rhs_nohat = rhs_hat.no_hat()

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

        # start_time = time.time()
        out_hat = rhs_hat.reconstruct_from_wavenumbers(solve_poisson_for_single_wavenumber)
        # print(str(time.time() - start_time) + " seconds used for reconstruction.")
        out = out_hat.no_hat()

        # u_ana.plot(out)

        tol = 1e-8
        # print(abs(u_ana - out))
        self.assertTrue(abs(u_ana - out) < tol)


    def test_poisson_no_slices(self):
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
        self.assertTrue(abs(u_ana - out) < tol)


    def test_navier_stokes_laminar(self, Ny=96, pertubation_factor=0.01):
        Re = 1.5e0

        end_time = 0.2
        NavierStokesVelVort.max_dt = 1e10
        nse = solve_navier_stokes_laminar(
            Re=Re,
            Nx=24,
            Ny=Ny,
            Nz=16,
            end_time=end_time,
            pertubation_factor=pertubation_factor,
        )
        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        nse.solve()

        vel_x_fn_ana = lambda X: -1 * nse.u_max_over_u_tau * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
        vel_x_ana = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana")


        print("Doing post-processing")
        for i in jnp.arange(nse.time_step)[-4:]:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            tol = 6e-5
            # print(abs(vel[0] - vel_x_ana))
            # print(abs(vel[1]))
            # print(abs(vel[2]))
            # check that the simulation is really converged
            self.assertTrue(abs(vel[0] - vel_x_ana) < tol)
            self.assertTrue(abs(vel[1]) < tol)
            self.assertTrue(abs(vel[2]) < tol)


    #TODO
    # def test_navier_stokes_laminar_convergence(self):
    #     Nys = [24, 48, 96]
    #     end_time = 10

    #     def run(Ny):
    #         nse = solve_navier_stokes_laminar(
    #             Re=1, end_time=end_time, Ny=Ny, pertubation_factor=0
    #         )
    #         nse.solve()
    #         vel_x_fn_ana = (
    #             lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    #         )
    #         # vel_x_ana = Field.FromFunc(nse.domain, vel_x_fn_ana, name="vel_x_ana")
    #         vel_hat = nse.get_latest_field("velocity_hat")
    #         vel = vel_hat.no_hat()
    #         return vel[0].l2error(vel_x_fn_ana)

    #     errors = list(map(run, Nys))
    #     errorsLog = list(map(lambda x: jnp.log2(x), errors))
    #     print(errors)

    #     def fittingFunc(x, a, b):
    #         return a + b * x

    #     result = optimization.curve_fit(fittingFunc, Nys, errorsLog)
    #     print(result)




    def test_linear_stability(self):
        n = 64
        # n = 4
        Re = 5772.22
        alpha = 1.02056

        lsc = LinearStabilityCalculation(Re, alpha, n)
        evs, _ = lsc.calculate_eigenvalues()
        # print(evs[0])
        # print(evecs[0])
        self.assertTrue(evs[0].real <= 0.0 and evs[0].real >= -1e-8)



    def test_pertubation_laminar(self, Ny=48, pertubation_factor=0.01):
        Re = 1.5e0

        end_time = 0.1
        nse = solve_navier_stokes_pertubation(
            Re=Re,
            Nx=16,
            Ny=Ny,
            Nz=16,
            end_time=end_time,
            pertubation_factor=pertubation_factor,
        )

        plot_interval = 1
        nse.before_time_step_fn = None
        nse.solve()

        print("Doing post-processing")
        for i in jnp.arange(nse.time_step)[-4:]:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            tol = 5e-7
            print(abs(vel[0]))
            print(abs(vel[1]))
            print(abs(vel[2]))
            # check that the simulation is really converged
            self.assertTrue(abs(vel[0]) < tol)
            self.assertTrue(abs(vel[1]) < tol)
            self.assertTrue(abs(vel[2]) < tol)


    def test_2d_growth(self):
        growth_5500 = run_pseudo_2d_pertubation(5500, 0.1)
        # growth_6000 = run_pseudo_2d_pertubation(6000, 0.3) # TODO tweak this until it works reliably
        growth_6500 = run_pseudo_2d_pertubation(6500, 0.2)
        # print("growth_5500: ", growth_5500)
        # print("growth_6000: ", growth_6000)
        # print("growth_6500: ", growth_6500)
        self.assertTrue(all([growth < 0 for growth in growth_5500]), "Expected pertubations to decay for Re=5500.")
        # self.assertTrue(all([growth > 0 for growth in growth_6000]), "Expected pertubations to increase for Re=6000.")
        self.assertTrue(all([growth > 0 for growth in growth_6500]), "Expected pertubations to increase for Re=6500.")

if __name__ == '__main__':
    unittest.main()

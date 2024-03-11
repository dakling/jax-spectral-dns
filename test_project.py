#!/usr/bin/env python3

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.figure as figure

# from importlib import reload
import sys

jax.config.update("jax_enable_x64", True)

# try:
#     reload(sys.modules["domain"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load Domain")
from domain import PhysicalDomain

# try:
#     reload(sys.modules["field"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load Field")
from field import Field, PhysicalField, FourierFieldSlice, VectorField

# try:
#     reload(sys.modules["equation"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load equation")
from equation import Equation

# try:
#     reload(sys.modules["navier_stokes"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load Navier Stokes")
from navier_stokes import NavierStokesVelVort, solve_navier_stokes_laminar

# try:
#     reload(sys.modules["navier_stokes_perturbation"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load navier-stokes-perturbation")
from navier_stokes_perturbation import solve_navier_stokes_perturbation

# try:
#     reload(sys.modules["linear_stability_calculation"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load linear stability")
from linear_stability_calculation import LinearStabilityCalculation

# try:
#     reload(sys.modules["examples"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load examples")
from examples import run_pseudo_2d_perturbation

NoneType = type(None)


class TestProject(unittest.TestCase):
    def setUp(self):
        Equation.initialize()

    def test_1D_cheb(self):
        Nx = 48
        domain = PhysicalDomain.create((Nx,), (False,))

        u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
        # u_fn = lambda X: 0.0 * jnp.cos(X[0] * jnp.pi / 2) + 2
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_1d_cheb")
        # u.update_boundary_conditions()
        u_x = u.diff(0, 1)
        u_xx = u.diff(0, 2)

        u_x_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.sin(X[0] * jnp.pi / 2) * jnp.pi / 2,
            name="u_x_ana",
        )
        u_xx_ana = PhysicalField.FromFunc(
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
        domain = PhysicalDomain.create((Nx,), (True,), scale_factors=(scale_factor,))

        u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_1d_periodic")
        u.update_boundary_conditions()
        u_x = u.diff(0, 1)
        u_xx = u.diff(0, 2)

        u_diff_fn = (
            lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor)
            * 2
            * jnp.pi
            / scale_factor
        )
        u_diff_fn_2 = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
            * (2 * jnp.pi / scale_factor) ** 2
        )
        u_x_ana = PhysicalField.FromFunc(domain, func=u_diff_fn, name="u_x_ana")
        u_xx_ana = PhysicalField.FromFunc(domain, func=u_diff_fn_2, name="u_xx_ana")

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
        domain = PhysicalDomain.create((Nx, Ny), (True, False), scale_factors=(scale_factor, 1.0))

        u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor) * jnp.cos(
            X[1] * jnp.pi / 2
        )
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_2d")
        u.update_boundary_conditions()
        u_x = u.diff(0, 1)
        u_xx = u.diff(0, 2)
        u_y = u.diff(1, 1)
        u_yy = u.diff(1, 2)

        u_x_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor)
            * jnp.cos(X[1] * jnp.pi / 2)
            * 2
            * jnp.pi
            / scale_factor,
            name="u_x_ana",
        )
        u_xx_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
            * jnp.cos(X[1] * jnp.pi / 2)
            * (2 * jnp.pi / scale_factor) ** 2,
            name="u_xx_ana",
        )
        u_y_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
            * jnp.pi
            / 2
            * jnp.sin(X[1] * jnp.pi / 2),
            name="u_y_ana",
        )
        u_yy_ana = PhysicalField.FromFunc(
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
        domain = PhysicalDomain.create(
            (Nx, Ny, Nz),
            (True, False, True),
            scale_factors=(scale_factor_x, 1.0, scale_factor_z),
        )

        u_fn = (
            lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
        )
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_3d")
        # u.update_boundary_conditions()
        u_x = u.diff(0, 1)
        u_xx = u.diff(0, 2)
        u_y = u.diff(1, 1)
        u_yy = u.diff(1, 2)
        u_z = u.diff(2, 1)
        u_zz = u.diff(2, 2)

        u_x_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
            * 2
            * jnp.pi
            / scale_factor_x,
            name="u_x_ana",
        )
        u_xx_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z)
            * (2 * jnp.pi / scale_factor_x) ** 2,
            name="u_xx_ana",
        )
        u_y_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.pi
            / 2
            * jnp.sin(X[1] * jnp.pi / 2)
            * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z),
            name="u_y_ana",
        )
        u_yy_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * (jnp.pi / 2) ** 2
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.cos(X[2] * 2 * jnp.pi / scale_factor_z),
            name="u_yy_ana",
        )
        u_z_ana = PhysicalField.FromFunc(
            domain,
            func=lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * jnp.pi / 2)
            * jnp.sin(X[2] * 2 * jnp.pi / scale_factor_z)
            * 2
            * jnp.pi
            / scale_factor_z,
            name="u_z_ana",
        )
        u_zz_ana = PhysicalField.FromFunc(
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
        domain = PhysicalDomain.create((Nx,), (True,), scale_factors=(scale_factor,))
        # domain = PhysicalDomain.create((Nx,), (True,))

        u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
        # u_fn = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / scale_factor))
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_1d")
        u_hat = u.hat()

        u_diff_fn = (
            lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor)
            * 2
            * jnp.pi
            / scale_factor
            # lambda X: jnp.cos(X[0]) / scale_factor * jnp.exp(jnp.sin(X[0] / scale_factor))
        )
        u_diff_ana = PhysicalField.FromFunc(domain, func=u_diff_fn, name="u_1d_diff_ana")

        u_diff_fn_2 = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
            * (2 * jnp.pi / scale_factor) ** 2
        )
        u_diff_ana_2 = PhysicalField.FromFunc(domain, func=u_diff_fn_2, name="u_1d_diff_2_ana")

        u_int_fn = lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor) * (
            2 * jnp.pi / scale_factor
        ) ** (-1)
        u_int_ana = PhysicalField.FromFunc(domain, func=u_int_fn, name="u_1d_int_ana")

        u_int_fn_2 = lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor) * (
            2 * jnp.pi / scale_factor
        ) ** (-2)
        u_int_ana_2 = PhysicalField.FromFunc(domain, func=u_int_fn_2, name="u_1d_int_2_ana")

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

        u_dom_diff = PhysicalField(domain, domain.diff(u.data, 0, 1))
        u_dom_diff_2 = PhysicalField(domain, domain.diff(u.data, 0, 2))
        u_dom_diff.name = "u_1d_dom_diff"
        u_dom_diff_2.name = "u_1d_dom_diff_2"
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
        self.assertTrue(abs(u_dom_diff - u_diff_ana) < tol)
        self.assertTrue(abs(u_dom_diff_2 - u_diff_ana_2) < tol)

    def test_fourier_2D(self):
        Nx = 24
        Ny = Nx + 4
        scale_factor_x = 1.0
        scale_factor_y = 2.0
        # scale_factor_x = 2.0 * jnp.pi
        # scale_factor_y = 2.0 * jnp.pi
        domain = PhysicalDomain.create(
            (Nx, Ny), (True, True), scale_factors=(scale_factor_x, scale_factor_y)
        )
        # domain = PhysicalDomain.create((Nx, Ny), (True, True))

        u_fn = lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x) * jnp.cos(
            X[1] * 2 * jnp.pi / scale_factor_y
        )
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_2d")

        u_x_fn = (
            lambda X: -jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * 2
            * jnp.pi
            / scale_factor_x
        )
        u_x_ana = PhysicalField.FromFunc(domain, func=u_x_fn, name="u_2d_x_ana")
        u_xx_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_x) ** 2
        )
        u_xx_ana = PhysicalField.FromFunc(domain, func=u_xx_fn, name="u_2d_xx_ana")

        u_y_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.sin(X[1] * 2 * jnp.pi / scale_factor_y)
            * 2
            * jnp.pi
            / scale_factor_y
        )
        u_y_ana = PhysicalField.FromFunc(domain, func=u_y_fn, name="u_2d_y_ana")
        u_yy_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_y) ** 2
        )
        u_yy_ana = PhysicalField.FromFunc(domain, func=u_yy_fn, name="u_2d_yy_ana")

        u_int_x_fn = (
            lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_x) ** (-1)
        )
        u_int_x_ana = PhysicalField.FromFunc(domain, func=u_int_x_fn, name="u_2d_int_x_ana")
        u_int_xx_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_x) ** (-2)
        )
        u_int_xx_ana = PhysicalField.FromFunc(domain, func=u_int_xx_fn, name="u_2d_int_xx_ana")

        u_int_y_fn = (
            lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.sin(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_y) ** (-1)
        )
        u_int_y_ana = PhysicalField.FromFunc(domain, func=u_int_y_fn, name="u_2d_int_y_ana")
        u_int_yy_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_y) ** (-2)
        )
        u_int_yy_ana = PhysicalField.FromFunc(domain, func=u_int_yy_fn, name="u_2d_int_yy_ana")

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
        domain = PhysicalDomain.create(
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
            lambda X: 1.0 * (jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x))
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
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_3d")
        v = PhysicalField.FromFunc(domain, func=u_0_fn, name="v_3d")
        w = PhysicalField.FromFunc(domain, func=u_0_fn, name="w_3d")
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
        domain = PhysicalDomain.create((Nx,), (False,))

        u_fn = lambda X: jnp.cos(X[0] * jnp.pi / 2)
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_1d")

        u_fn_int = lambda X: -((2 / jnp.pi) ** 2) * jnp.cos(X[0] * jnp.pi / 2)
        u_int_ana = PhysicalField.FromFunc(domain, func=u_fn_int, name="u_1d_int_ana")
        u_int = u.integrate(0, order=2, bc_left=0.0, bc_right=0.0)
        u_int.name = "u_int"
        # u_int.plot(u, u_int_ana)

        tol = 1e-7
        # print(abs(u_int - u_int_ana))
        self.assertTrue(abs(u_int - u_int_ana) < tol)

    def test_cheb_integration_2D(self):
        Nx = 24
        Ny = Nx + 4
        domain = PhysicalDomain.create((Nx, Ny), (True, False))

        u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_2d")

        u_fn_int_1 = lambda X: jnp.cos(X[0]) * (2 / jnp.pi) * jnp.sin(
            X[1] * jnp.pi / 2
        ) + (2 / jnp.pi) * jnp.cos(X[0])
        u_int_1_ana = PhysicalField.FromFunc(domain, func=u_fn_int_1, name="u_2d_int_1_ana")
        u_fn_int_2 = (
            lambda X: -jnp.cos(X[0]) * (2 / jnp.pi) ** 2 * jnp.cos(X[1] * jnp.pi / 2)
        )
        u_int_2_ana = PhysicalField.FromFunc(domain, func=u_fn_int_2, name="u_2d_int_2_ana")
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
        domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True))

        u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_3d")

        u_fn_int = (
            lambda X: -jnp.cos(X[0])
            * jnp.cos(X[2])
            * (2 / jnp.pi) ** 2
            * jnp.cos(X[1] * jnp.pi / 2)
        )
        u_int_ana = PhysicalField.FromFunc(domain, func=u_fn_int, name="u_3d_int_ana")
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
        domain_1D_fourier = PhysicalDomain.create((Nx,), (True,), scale_factors=(sc_x,))
        u_fn_1d_fourier = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x))
        u_1d_fourier = PhysicalField.FromFunc(
            domain_1D_fourier, u_fn_1d_fourier, name="u_1d_fourier"
        )
        self.assertTrue(
            abs(u_1d_fourier.definite_integral(0) - 1.2660658777520084) < tol
        )
        # Chebyshev
        domain_1D_cheb = PhysicalDomain.create((Nx,), (False,))
        u_fn_1d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - 1
        u_1d_cheb = PhysicalField.FromFunc(domain_1D_cheb, u_fn_1d_cheb, name="u_1d_cheb")
        # print(u_1d_cheb.definite_integral(0))
        # print(abs(u_1d_cheb.definite_integral(0) - 0.5321317555))
        self.assertTrue(abs(u_1d_cheb.definite_integral(0) - 0.5321317555) < tol)
        # 2D
        # Fourier
        Ny = 64
        sc_y = 2.0
        domain_2D_fourier = PhysicalDomain.create((Nx, Ny), (True, True), scale_factors=(sc_x, sc_y))
        u_fn_2d_fourier = lambda X: jnp.exp(
            jnp.sin(X[0] * 2 * jnp.pi / sc_x)
        ) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y) ** 2)
        u_2d_fourier = PhysicalField.FromFunc(
            domain_2D_fourier, u_fn_2d_fourier, name="u_2d_fourier"
        )
        # print(abs(u_2d_fourier.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        self.assertTrue(
            (
                abs(
                    u_2d_fourier.definite_integral(1).definite_integral(0)
                    - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
                )
            )
            < tol
        )
        self.assertTrue(
            (
                abs(
                    u_2d_fourier.volume_integral()
                    - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
                )
            )
            < tol
        )
        # Chebyshev
        domain_2D_cheb = PhysicalDomain.create((Nx, Ny), (False, False))
        u_fn_2d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - jnp.exp(
            jnp.sin(X[1] * 2 * jnp.pi / sc_y) ** 2
        )
        u_2d_cheb = PhysicalField.FromFunc(domain_2D_cheb, u_fn_2d_cheb, name="u_2d_cheb")
        # print(abs(u_2d_cheb.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        # print((u_2d_cheb.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
        self.assertTrue(
            (
                abs(
                    u_2d_cheb.definite_integral(1).definite_integral(0)
                    - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
                )
            )
            < tol
        )
        self.assertTrue(
            (
                abs(
                    u_2d_cheb.volume_integral()
                    - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
                )
            )
            < tol
        )
        # Mixed
        domain_2D_mixed = PhysicalDomain.create((Nx, Ny), (False, True), scale_factors=(1.0, sc_y))
        u_2d_mixed = PhysicalField.FromFunc(domain_2D_mixed, u_fn_2d_cheb, name="u_2d_mixed")
        # print(abs(u_2d_mixed.definite_integral(1).definite_integral(0) - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962))
        self.assertTrue(
            (
                abs(
                    u_2d_mixed.definite_integral(1).definite_integral(0)
                    - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
                )
            )
            < tol
        )
        self.assertTrue(
            (
                abs(
                    u_2d_mixed.volume_integral()
                    - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
                )
            )
            < tol
        )
        domain_2D_mixed_2 = PhysicalDomain.create((Nx, Ny), (True, False), scale_factors=(sc_x, 1.0))
        u_2d_mixed_2 = PhysicalField.FromFunc(
            domain_2D_mixed_2, u_fn_2d_cheb, name="u_2d_mixed_2"
        )
        # print(abs(u_2d_mixed_2.definite_integral(1).definite_integral(0) - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812))
        self.assertTrue(
            (
                abs(
                    u_2d_mixed_2.definite_integral(1).definite_integral(0)
                    - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
                )
            )
            < tol
        )
        self.assertTrue(
            (
                abs(
                    u_2d_mixed_2.volume_integral()
                    - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
                )
            )
            < tol
        )
        # 3D
        # Fourier
        # Nx = 96
        # Ny = 96
        Nz = 96
        sc_z = 3.0
        domain_3D_fourier = PhysicalDomain.create(
            (Nx, Ny, Nz), (True, True, True), scale_factors=(sc_x, sc_y, sc_z)
        )
        u_fn_3d_fourier = lambda X: jnp.exp(
            jnp.sin(X[0] * 2 * jnp.pi / sc_x)
        ) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y) ** 2) * jnp.exp(
            jnp.cos(X[2] * 2 * jnp.pi / sc_z) ** 2
        )
        u_3d_fourier = PhysicalField.FromFunc(
            domain_3D_fourier, u_fn_3d_fourier, name="u_3d_fourier"
        )
        # print(u_3d_fourier.definite_integral(2).definite_integral(1).definite_integral(0) - -10.84981433261992)
        self.assertTrue(
            (
                abs(
                    u_3d_fourier.definite_integral(2)
                    .definite_integral(1)
                    .definite_integral(0)
                    - -10.84981433261992
                )
            )
            < tol
        )
        self.assertTrue(
            (abs(u_3d_fourier.volume_integral() - -10.84981433261992)) < tol
        )
        # Chebyshev
        domain_3D_cheb = PhysicalDomain.create((Nx, Ny, Nz), (False, False, False))
        u_fn_3d_cheb = (
            lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi))
            - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi) ** 2)
            + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi) ** 2)
        )
        u_3d_cheb = PhysicalField.FromFunc(domain_3D_cheb, u_fn_3d_cheb, name="u_3d_cheb")
        # print(u_3d_cheb.definite_integral(2).definite_integral(1).definite_integral(0) - 10.128527022082872)
        self.assertTrue(
            (
                abs(
                    u_3d_cheb.definite_integral(2)
                    .definite_integral(1)
                    .definite_integral(0)
                    - 10.128527022082872
                )
            )
            < tol
        )
        self.assertTrue((abs(u_3d_cheb.volume_integral() - 10.128527022082872)) < tol)
        # Mixed
        domain_3D_mixed = PhysicalDomain.create(
            (Nx, Ny, Nz), (True, False, True), scale_factors=(sc_x, 1.0, sc_z)
        )
        u_fn_3d_mixed = (
            lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x))
            - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi) ** 2)
            + jnp.exp(jnp.cos(X[2] * 2 * jnp.pi / sc_z) ** 2)
        )
        u_3d_mixed = PhysicalField.FromFunc(domain_3D_mixed, u_fn_3d_mixed, name="u_3d_mixed")
        # print(u_3d_mixed.definite_integral(2).definite_integral(1).definite_integral(0) - 7.596395266449558)
        self.assertTrue(
            (
                abs(
                    u_3d_mixed.definite_integral(2)
                    .definite_integral(1)
                    .definite_integral(0)
                    - 7.596395266449558
                )
            )
            < tol
        )
        self.assertTrue((abs(u_3d_mixed.volume_integral() - 7.596395266449558)) < tol)

    def test_poisson_slices(self):
        Nx = 24
        Ny = Nx + 4
        Nz = Nx - 4
        # Nx = 4
        # Ny = 4
        # Nz = 6
        scale_factor_x = 1.0
        scale_factor_z = 1.0

        domain = PhysicalDomain.create(
            (Nx, Ny, Nz),
            (True, False, True),
            scale_factors=(scale_factor_x, 1.0, scale_factor_z),
        )
        domain_y = PhysicalDomain.create((Ny,), (False,))

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
        rhs = PhysicalField.FromFunc(domain, rhs_fn, name="rhs")

        u_ana_fn = (
            lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
            * jnp.cos(X[1] * jnp.pi / 2)
        )
        u_ana = PhysicalField.FromFunc(domain, u_ana_fn, name="u_ana")
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
                rhs_hat.fourier_domain.grid[0][kx],
                rhs_hat.fourier_domain.grid[2][kz],
                ks_int=[kx, kz],
            )
            out = rhs_hat_slice.solve_poisson(mat)
            # out = rhs_hat_slice.solve_poisson()
            return out.data

        # start_time = time.time()
        out_hat = rhs_hat.reconstruct_from_wavenumbers(
            solve_poisson_for_single_wavenumber
        )
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
        domain = PhysicalDomain.create(
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
        rhs = PhysicalField.FromFunc(domain, rhs_fn, name="rhs")

        u_ana_fn = (
            lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.sin((X[2] + 1.0) * 2 * jnp.pi / scale_factor_z)
            * jnp.cos(X[1] * jnp.pi / 2)
        )

        u_ana = PhysicalField.FromFunc(domain, u_ana_fn, name="u_ana")
        rhs_hat = rhs.hat()

        out_hat = rhs_hat.solve_poisson()
        out = out_hat.no_hat()

        # u_ana.plot(out)

        tol = 1e-8
        # print(abs(u_ana - out))
        self.assertTrue(abs(u_ana - out) < tol)

    def test_navier_stokes_laminar(self, Ny=96, perturbation_factor=0.01):
        for activate_jit in [True, False]:
            Re = 1.5e0

            end_time = 1.0
            nse = solve_navier_stokes_laminar(
                Re=Re,
                Nx=24,
                Ny=Ny,
                Nz=16,
                end_time=end_time,
                # max_dt=1e-2,
                dt=2e-2,
                perturbation_factor=perturbation_factor,
            )
            def before_time_step(nse):
                u = nse.get_latest_field("velocity_hat").no_hat()
                u.set_time_step(nse.time_step)
                u.plot_3d(2)

            Equation.initialize()

            if activate_jit:
                nse.activate_jit()
                nse.before_time_step_fn = None
            else:
                nse.deactivate_jit()
                nse.before_time_step_fn = before_time_step
            nse.after_time_step_fn = None
            nse.solve()

            vel_x_fn_ana = (
                lambda X: -1 * nse.get_u_max_over_u_tau() * (X[1] + 1) * (X[1] - 1)
                + 0.0 * X[0] * X[2]
            )
            vel_x_ana = PhysicalField.FromFunc(nse.get_physical_domain() , vel_x_fn_ana, name="vel_x_ana")

            print("Doing post-processing")
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            tol = 6e-5
            print(abs(vel[0] - vel_x_ana))
            print(abs(vel[1]))
            print(abs(vel[2]))
            # check that the simulation is really converged
            self.assertTrue(abs(vel[0] - vel_x_ana) < tol)
            self.assertTrue(abs(vel[1]) < tol)
            self.assertTrue(abs(vel[2]) < tol)

    # TODO
    # def test_navier_stokes_laminar_convergence(self):
    #     Nys = [24, 48, 96]
    #     end_time = 10

    #     def run(Ny):
    #         nse = solve_navier_stokes_laminar(
    #             Re=1, end_time=end_time, Ny=Ny, perturbation_factor=0
    #         )
    #         nse.solve()
    #         vel_x_fn_ana = (
    #             lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    #         )
    #         # vel_x_ana = PhysicalField.FromFunc(nse.domain, vel_x_fn_ana, name="vel_x_ana")
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
        n = 50
        # n = 4
        Re = 5772.22
        alpha = 1.02056

        lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=n)
        evs, _ = lsc.calculate_eigenvalues()
        # evs, evecs = lsc.calculate_eigenvalues()
        # print(evs[0])
        # print(evecs[0])
        self.assertTrue(evs[0].real <= 0.0 and evs[0].real >= -1e-8)

    def test_perturbation_laminar(self, Ny=48, perturbation_factor=0.01):
        for activate_jit in [True, False]:
            Re = 1.5e0

            end_time = 1.0
            nse = solve_navier_stokes_perturbation(
                Re=Re,
                Nx=16,
                Ny=Ny,
                Nz=16,
                end_time=end_time,
                dt=2e-2,
                perturbation_factor=perturbation_factor,
            )

            if activate_jit:
                nse.activate_jit()
            else:
                nse.deactivate_jit()
            nse.before_time_step_fn = None
            nse.solve()

            print("Doing post-processing")
            vel_hat = nse.get_latest_field("velocity_hat")
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
        growth_5500_data = run_pseudo_2d_perturbation(Re=5500, end_time=1.5, eps=1e-1, linearize=True, Nx=50, Ny=90, Nz=2)
        growth_6000_data = run_pseudo_2d_perturbation(Re=6000, end_time=1.5, eps=1e-1, linearize=True, Nx=50, Ny=90, Nz=2)
        growth_5500 = []
        growth_6000 = []
        for i in range(3):
            growth_5500.append(growth_5500_data[i][-1] - growth_5500_data[i][-2])
            growth_6000.append(growth_6000_data[i][-1] - growth_6000_data[i][-2])
        # print("growth_5500: ", growth_5500)
        # print("growth_6000: ", growth_6000)
        self.assertTrue(
            all([growth < 0 for growth in growth_5500]),
            "Expected perturbations to decay for Re=5500.",
        )
        self.assertTrue(
            all([growth > 0 for growth in growth_6000]),
            "Expected perturbations to increase for Re=6000.",
        )
        vel_final_jit_5500 = growth_5500_data[-1]
        vel_final_jit_6000 = growth_6000_data[-1]

        # Now check that the same result is obtained when using solve_scan. To
        # simplify and strengthen the test, only compare the final velocity
        # fields.
        Field.activate_jit_ = True
        data_no_jit_5500 = run_pseudo_2d_perturbation(Re=5500, end_time=1.5, eps=1e-1, linearize=True, Nx=50, Ny=90, Nz=2)
        data_no_jit_6000 = run_pseudo_2d_perturbation(Re=6000, end_time=1.5, eps=1e-1, linearize=True, Nx=50, Ny=90, Nz=2)
        vel_final_no_jit_5500 = data_no_jit_5500[-1]
        vel_final_no_jit_6000 = data_no_jit_6000[-1]

        print((vel_final_jit_5500 - vel_final_no_jit_5500).energy())
        print((vel_final_jit_6000 - vel_final_no_jit_6000).energy())
        self.assertTrue(
            (vel_final_jit_5500 - vel_final_no_jit_5500).energy() < 1e-8
        )
        self.assertTrue(
            (vel_final_jit_6000 - vel_final_no_jit_6000).energy() < 1e-8
        )

    def test_2d_growth_rates_quantitatively(self):

        def run_re(Re, rotated=False):
            if rotated:
                return run_pseudo_2d_perturbation(
                    Re=Re, end_time=1e0, Nx=6, Ny=64, Nz=2, linearize=True, plot=True, save=False, eps=1.0, dt=1e-2, rotated=True, aliasing=1.0
                )
            else:
                return run_pseudo_2d_perturbation(
                    Re=Re, end_time=1e0, Nx=2, Ny=64, Nz=6, linearize=True, plot=True, save=False, eps=1.0, dt=1e-2, aliasing=1.0
                )


        def run(rotated=False):
            ts = []
            energy = []
            energy_ana = []
            for Re in [5500, 5772.22, 6000]:
                out = run_re(Re, rotated)
                ts.append(out[-2])
                energy.append(out[0])
                energy_ana.append(out[3])
            return (ts, jnp.array(energy), jnp.array(energy_ana))

        def calculate_growth_rates(ts, energy, energy_ana):
            start_index = 1 # don't start at 0 to allow for some initial transient
            time = ts[0][-1] - ts[0][start_index]
            print("Re = 5500:")
            growth_rate = (energy[0][-1] - energy[0][start_index]) / (time * energy_ana[0][start_index])
            growth_rate_ana = (energy_ana[0][-1] - energy_ana[0][start_index]) / (time * energy_ana[0][start_index])
            print("growth rate: ", growth_rate)
            print("growth rate (analytical): ", growth_rate_ana)
            print("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_5500 = abs((growth_rate_ana - growth_rate) / (0.5 * (growth_rate_ana + growth_rate)))
            print("relative error: ", rel_error_5500)
            assert rel_error_5500 < 1e-2

            print("Re = 5772.22:")
            growth_rate = (energy[1][-1] - energy[1][start_index]) / (time * energy_ana[1][start_index])
            growth_rate_ana = (energy_ana[1][-1] - energy_ana[1][start_index]) / (time * energy_ana[1][start_index])
            print("growth rate: ", growth_rate)
            print("growth rate (analytical): ", growth_rate_ana)
            print("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_5772 = abs((growth_rate_ana - growth_rate) / (0.5 * (growth_rate_ana + growth_rate)))
            print("relative error: ", rel_error_5772)
            assert rel_error_5772 < 3 # (this can be quite large as the denominator is almost zero)

            print("Re = 6000:")
            growth_rate = (energy[2][-1] - energy[2][start_index]) / (time * energy_ana[2][start_index])
            growth_rate_ana = (energy_ana[2][-1] - energy_ana[2][start_index]) / (time * energy_ana[2][start_index])
            print("growth rate: ", growth_rate)
            print("growth rate (analytical): ", growth_rate_ana)
            print("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_6000 = abs((growth_rate_ana - growth_rate) / (0.5 * (growth_rate_ana + growth_rate)))
            print("relative error: ", rel_error_6000)
            assert rel_error_6000 < 1e-2



        def plot(ts, dataset, dataset_ana):
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            ax.plot(ts[0], dataset_ana[0]/dataset_ana[0][0], "-", label="Re_5500")
            ax.plot(ts[0], dataset[0]/dataset_ana[0][0], ".", label="Re_5500")
            ax.plot(ts[1], dataset_ana[1]/dataset_ana[1][0], "-", label="Re_5772")
            ax.plot(ts[1], dataset[1]/dataset_ana[1][0], ".", label="Re_5772")
            ax.plot(ts[2], dataset_ana[2]/dataset_ana[2][0], "-", label="Re_6000")
            ax.plot(ts[2], dataset[2]/dataset_ana[2][0], ".", label="Re_6000")
            fig.legend()
            fig.savefig(
                "plots/" + "energy" + ".png"
            )

        def main():
            for rotated in [False, True]:
                print("testing growth rates in " + ("rotated" if rotated else "normal") + " domain")
                ts, energy, energy_ana = run(rotated)
                plot(ts, energy, energy_ana)
                calculate_growth_rates(ts, energy, energy_ana)

        main()


    # def test_timesteppers(self):
    #     Re = 6000
    #     Nx = 4
    #     Ny = 50
    #     Nz = 4
    #     end_time = 1e-10
    #     linearize = False
    #     alpha = 1.02056

    #     lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=50)
    #     nse_rk = solve_navier_stokes_perturbation(
    #         Re=Re,
    #         Nx=Nx,
    #         Ny=Ny,
    #         Nz=Nz,
    #         end_time=end_time,
    #         perturbation_factor=0.0,
    #         scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-6),
    #     )
    #     nse_rk.set_linearize(linearize)
    #     nse_rk.max_dt = 1e-3

    #     U = lsc.velocity_field(nse_rk.domain_no_hat)
    #     U.plot_3d(2)
    #     # raise Exception("break")
    #     eps_ = 1.0 * jnp.sqrt(U.energy())
    #     U_hat = U.hat()
    #     nse_rk.init_velocity(U_hat * eps_)
    #     nse_rk.prepare()
    #     vel_rk = nse_rk.get_latest_field("velocity_hat").no_hat()

    #     nse_cnab = solve_navier_stokes_perturbation(
    #         Re=Re,
    #         Nx=Nx,
    #         Ny=Ny,
    #         Nz=Nz,
    #         end_time=end_time,
    #         perturbation_factor=0.0,
    #         scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-6),
    #     )
    #     nse_cnab.set_linearize(linearize)
    #     nse_cnab.max_dt = 1e-3
    #     nse_cnab.init_velocity(U_hat * eps_)
    #     nse_cnab.prepare()
    #     vel_cnab = nse_cnab.get_latest_field("velocity_hat").no_hat()

    #     vel_diff = vel_rk - vel_cnab
    #     print(abs(vel_diff))
    #     assert abs(vel_diff) <  1e-10

    #     nse_rk.perform_runge_kutta_step()
    #     nse_cnab.perform_cn_ab_step()

    #     vel_rk = nse_rk.get_latest_field("velocity_hat").no_hat()
    #     vel_cnab = nse_cnab.get_latest_field("velocity_hat").no_hat()
    #     vel_rk.time_step = 1
    #     vel_cnab.time_step = 1

    #     vel_diff = vel_rk - vel_cnab
    #     for i in range(3):
    #         vel_diff[i].name = "velocity_difference_" + "xyz"[i]
    #         vel_rk[i].name = "velocity_rk_" + "xyz"[i]
    #         vel_cnab[i].name = "velocity_cnab_" + "xyz"[i]
    #         vel_rk[i].plot_3d(2)
    #         vel_cnab[i].plot_3d(2)
    #         vel_diff[i].plot_3d(2)
    #     print(abs(vel_diff))
    #     assert abs(vel_diff) < 1e-8

    #     nse_rk.perform_runge_kutta_step()
    #     nse_cnab.perform_cn_ab_step()

    #     vel_rk = nse_rk.get_latest_field("velocity_hat").no_hat()
    #     vel_cnab = nse_cnab.get_latest_field("velocity_hat").no_hat()
    #     vel_rk.time_step = 2
    #     vel_cnab.time_step = 2

    #     vel_diff = vel_rk - vel_cnab
    #     # for i in range(3):
    #     #     vel_diff[i].name = "velocity_difference_" + "xyz"[i]
    #     #     vel_cnab[i].plot_3d(2)
    #     #     vel_diff[i].plot_3d(2)
    #     print(abs(vel_diff))
    #     assert abs(vel_diff) < 1e-8

if __name__ == "__main__":
    unittest.main()

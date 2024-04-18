#!/usr/bin/env python3
from __future__ import annotations

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.figure as figure
from matplotlib.axes import Axes

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from typing import TYPE_CHECKING, cast

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, PhysicalField, FourierFieldSlice, VectorField
from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.navier_stokes import (
    NavierStokesVelVort,
    solve_navier_stokes_laminar,
)
from jax_spectral_dns.navier_stokes_perturbation import solve_navier_stokes_perturbation
from jax_spectral_dns.linear_stability_calculation import LinearStabilityCalculation
from jax_spectral_dns.examples import (
    run_pseudo_2d_perturbation,
    run_transient_growth,
    run_transient_growth_nonpert,
)
from jax_spectral_dns._typing import np_float_array, jnp_array, np_jnp_array, jsd_float

if TYPE_CHECKING:
    from jax_spectral_dns._typing import pseudo_2d_perturbation_return_type

NoneType = type(None)


class TestProject(unittest.TestCase):
    def setUp(self) -> None:
        Equation.initialize()
        Equation.verbosity_level = 0  # suppress output

    def test_1D_cheb(self) -> None:
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
        print_verb("u", verbosity_level=3)
        print_verb(u, verbosity_level=3)
        print_verb("ux", verbosity_level=3)
        print_verb(u_x, verbosity_level=3)
        # u.plot_center(0, u_x, u_xx)
        tol = 5e-4
        print_verb(abs(u_x - u_x_ana), verbosity_level=3)
        print_verb(abs(u_xx - u_xx_ana), verbosity_level=3)
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)

    def test_1D_periodic(self) -> None:
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
        print_verb(abs(u_x - u_x_ana), verbosity_level=3)
        print_verb(abs(u_xx - u_xx_ana), verbosity_level=3)
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)

    def test_2D(self) -> None:
        Nx = 20
        # Ny = Nx
        Ny = 24
        scale_factor = 1.0
        domain = PhysicalDomain.create(
            (Nx, Ny), (True, False), scale_factors=(scale_factor, 1.0)
        )

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
        print_verb(abs(u_x - u_x_ana), verbosity_level=3)
        print_verb(abs(u_xx - u_xx_ana), verbosity_level=3)
        print_verb(abs(u_y - u_y_ana), verbosity_level=3)
        print_verb(abs(u_yy - u_yy_ana), verbosity_level=3)
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_yy - u_yy_ana) < tol)

    def test_3D(self) -> None:
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

        print_verb(abs(u_x - u_x_ana), verbosity_level=3)
        print_verb(abs(u_xx - u_xx_ana), verbosity_level=3)
        print_verb(abs(u_y - u_y_ana), verbosity_level=3)
        print_verb(abs(u_yy - u_yy_ana), verbosity_level=3)
        print_verb(abs(u_z - u_z_ana), verbosity_level=3)
        print_verb(abs(u_zz - u_zz_ana), verbosity_level=3)
        tol = 5e-5
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_xx - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_yy - u_yy_ana) < tol)
        self.assertTrue(abs(u_z - u_z_ana) < tol)
        self.assertTrue(abs(u_zz - u_zz_ana) < tol)

    def test_fourier_1D(self) -> None:
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
        u_diff_ana = PhysicalField.FromFunc(
            domain, func=u_diff_fn, name="u_1d_diff_ana"
        )

        u_diff_fn_2 = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor)
            * (2 * jnp.pi / scale_factor) ** 2
        )
        u_diff_ana_2 = PhysicalField.FromFunc(
            domain, func=u_diff_fn_2, name="u_1d_diff_2_ana"
        )

        u_int_fn = lambda X: jnp.sin(X[0] * 2 * jnp.pi / scale_factor) * (
            2 * jnp.pi / scale_factor
        ) ** (-1)
        u_int_ana = PhysicalField.FromFunc(domain, func=u_int_fn, name="u_1d_int_ana")

        u_int_fn_2 = lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor) * (
            2 * jnp.pi / scale_factor
        ) ** (-2)
        u_int_ana_2 = PhysicalField.FromFunc(
            domain, func=u_int_fn_2, name="u_1d_int_2_ana"
        )

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
        print_verb(abs(u_int - u_int_ana), verbosity_level=3)
        print_verb(abs(u_int_2 - u_int_ana_2), verbosity_level=3)
        print_verb(abs(u_diff - u_diff_ana), verbosity_level=3)
        print_verb(abs(u_diff_2 - u_diff_ana_2), verbosity_level=3)
        self.assertTrue(abs(u_int - u_int_ana) < tol)
        self.assertTrue(abs(u_int_2 - u_int_ana_2) < tol)
        self.assertTrue(abs(u_diff - u_diff_ana) < tol)
        self.assertTrue(abs(u_diff_2 - u_diff_ana_2) < tol)
        self.assertTrue(abs(u_dom_diff - u_diff_ana) < tol)
        self.assertTrue(abs(u_dom_diff_2 - u_diff_ana_2) < tol)

    def test_fourier_2D(self) -> None:
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
        u_int_x_ana = PhysicalField.FromFunc(
            domain, func=u_int_x_fn, name="u_2d_int_x_ana"
        )
        u_int_xx_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_x) ** (-2)
        )
        u_int_xx_ana = PhysicalField.FromFunc(
            domain, func=u_int_xx_fn, name="u_2d_int_xx_ana"
        )

        u_int_y_fn = (
            lambda X: jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.sin(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_y) ** (-1)
        )
        u_int_y_ana = PhysicalField.FromFunc(
            domain, func=u_int_y_fn, name="u_2d_int_y_ana"
        )
        u_int_yy_fn = (
            lambda X: -jnp.cos(X[0] * 2 * jnp.pi / scale_factor_x)
            * jnp.cos(X[1] * 2 * jnp.pi / scale_factor_y)
            * (2 * jnp.pi / scale_factor_y) ** (-2)
        )
        u_int_yy_ana = PhysicalField.FromFunc(
            domain, func=u_int_yy_fn, name="u_2d_int_yy_ana"
        )

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
        print_verb(abs(u_x - u_x_ana), verbosity_level=3)
        print_verb(abs(u_x_2 - u_xx_ana), verbosity_level=3)
        print_verb(abs(u_y - u_y_ana), verbosity_level=3)
        print_verb(abs(u_y_2 - u_yy_ana), verbosity_level=3)
        print_verb(abs(u_int_x - u_int_x_ana), verbosity_level=3)
        print_verb(abs(u_int_xx - u_int_xx_ana), verbosity_level=3)
        print_verb(abs(u_int_y - u_int_y_ana), verbosity_level=3)
        print_verb(abs(u_int_yy - u_int_yy_ana), verbosity_level=3)
        self.assertTrue(abs(u_x - u_x_ana) < tol)
        self.assertTrue(abs(u_x_2 - u_xx_ana) < tol)
        self.assertTrue(abs(u_y - u_y_ana) < tol)
        self.assertTrue(abs(u_y_2 - u_yy_ana) < tol)
        self.assertTrue(abs(u_int_x - u_int_x_ana) < tol)
        self.assertTrue(abs(u_int_xx - u_int_xx_ana) < tol)
        self.assertTrue(abs(u_int_y - u_int_y_ana) < tol)
        self.assertTrue(abs(u_int_yy - u_int_yy_ana) < tol)

    def test_fourier_simple_3D(self) -> None:
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
        # print_verb(abs(U - U_nohat), verbosity_level=3)
        print_verb(abs(u_hat_x.no_hat() - u_x), verbosity_level=3)
        print_verb(abs(u_hat_xx.no_hat() - u_xx), verbosity_level=3)
        print_verb(abs(u_hat_y.no_hat() - u_y), verbosity_level=3)
        print_verb(abs(u_hat_yy.no_hat() - u_yy), verbosity_level=3)
        print_verb(abs(u_hat_z.no_hat() - u_z), verbosity_level=3)
        print_verb(abs(u_hat_zz.no_hat() - u_zz), verbosity_level=3)
        # print_verb(abs(u_int_hat_x.no_hat() - u_int_x), verbosity_level=3)
        # print_verb(abs(u_int_hat_xx.no_hat() - u_int_xx), verbosity_level=3)
        # print_verb(abs(u_int_hat_yy.no_hat() - u_int_yy), verbosity_level=3)
        # print_verb(abs(u_int_hat_z.no_hat() - u_int_z), verbosity_level=3)
        # print_verb(abs(u_int_hat_zz.no_hat() - u_int_zz), verbosity_level=3)
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

    def test_cheb_integration_1D(self) -> None:
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
        print_verb(abs(u_int - u_int_ana), verbosity_level=3)
        self.assertTrue(abs(u_int - u_int_ana) < tol)

    def test_cheb_integration_2D(self) -> None:
        Nx = 24
        Ny = Nx + 4
        domain = PhysicalDomain.create((Nx, Ny), (True, False))

        u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1] * jnp.pi / 2)
        u = PhysicalField.FromFunc(domain, func=u_fn, name="u_2d")

        u_fn_int_1 = lambda X: jnp.cos(X[0]) * (2 / jnp.pi) * jnp.sin(
            X[1] * jnp.pi / 2
        ) + (2 / jnp.pi) * jnp.cos(X[0])
        u_int_1_ana = PhysicalField.FromFunc(
            domain, func=u_fn_int_1, name="u_2d_int_1_ana"
        )
        u_fn_int_2 = (
            lambda X: -jnp.cos(X[0]) * (2 / jnp.pi) ** 2 * jnp.cos(X[1] * jnp.pi / 2)
        )
        u_int_2_ana = PhysicalField.FromFunc(
            domain, func=u_fn_int_2, name="u_2d_int_2_ana"
        )
        u_int_1 = u.integrate(1, order=1, bc_left=0.0)
        u_int_2 = u.integrate(1, order=2, bc_left=0.0, bc_right=0.0)
        u_int_1.name = "u_int_1"
        u_int_2.name = "u_int_2"
        # u_int_1.plot(u, u_int_1_ana)
        # u_int_2.plot(u, u_int_2_ana)

        tol = 1e-7
        print_verb(abs(u_int_1 - u_int_1_ana), verbosity_level=3)
        print_verb(abs(u_int_2 - u_int_2_ana), verbosity_level=3)
        self.assertTrue(abs(u_int_1 - u_int_1_ana) < tol)
        self.assertTrue(abs(u_int_2 - u_int_2_ana) < tol)

    def test_cheb_integration_3D(self) -> None:
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
        print_verb(abs(u_int - u_int_ana), verbosity_level=3)
        self.assertTrue(abs(u_int - u_int_ana) < tol)

    def test_definite_integral(self) -> None:
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
            abs(cast(float, u_1d_fourier.definite_integral(0)) - 1.2660658777520084)
            < tol
        )
        # Chebyshev
        domain_1D_cheb = PhysicalDomain.create((Nx,), (False,))
        u_fn_1d_cheb = lambda X: jnp.exp(jnp.sin(X[0] * 2 * jnp.pi / sc_x)) - 1
        u_1d_cheb = PhysicalField.FromFunc(
            domain_1D_cheb, u_fn_1d_cheb, name="u_1d_cheb"
        )
        print_verb(u_1d_cheb.definite_integral(0), verbosity_level=3)
        print_verb(
            abs(cast(float, u_1d_cheb.definite_integral(0)) - 0.5321317555),
            verbosity_level=3,
        )
        self.assertTrue(
            abs(cast(float, u_1d_cheb.definite_integral(0)) - 0.5321317555) < tol
        )
        # 2D
        # Fourier
        Ny = 64
        sc_y = 2.0
        domain_2D_fourier = PhysicalDomain.create(
            (Nx, Ny), (True, True), scale_factors=(sc_x, sc_y)
        )
        u_fn_2d_fourier = lambda X: jnp.exp(
            jnp.sin(X[0] * 2 * jnp.pi / sc_x)
        ) - jnp.exp(jnp.sin(X[1] * 2 * jnp.pi / sc_y) ** 2)
        u_2d_fourier = PhysicalField.FromFunc(
            domain_2D_fourier, u_fn_2d_fourier, name="u_2d_fourier"
        )
        print_verb(
            abs(
                cast(
                    float,
                    cast(
                        PhysicalField, u_2d_fourier.definite_integral(1)
                    ).definite_integral(0),
                )
                - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
            ),
            verbosity_level=3,
        )
        self.assertTrue(
            (
                abs(
                    cast(
                        float,
                        cast(
                            PhysicalField, u_2d_fourier.definite_integral(1)
                        ).definite_integral(0),
                    )
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
        u_2d_cheb = PhysicalField.FromFunc(
            domain_2D_cheb, u_fn_2d_cheb, name="u_2d_cheb"
        )
        print_verb(
            abs(
                cast(
                    float,
                    cast(
                        PhysicalField, u_2d_cheb.definite_integral(1)
                    ).definite_integral(0),
                )
                - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
            ),
            verbosity_level=3,
        )
        print_verb(
            (
                cast(
                    float,
                    cast(
                        PhysicalField, u_2d_cheb.definite_integral(1)
                    ).definite_integral(0),
                )
                - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
            ),
            verbosity_level=3,
        )
        self.assertTrue(
            (
                abs(
                    cast(
                        float,
                        cast(
                            PhysicalField, u_2d_cheb.definite_integral(1)
                        ).definite_integral(0),
                    )
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
        domain_2D_mixed = PhysicalDomain.create(
            (Nx, Ny), (False, True), scale_factors=(1.0, sc_y)
        )
        u_2d_mixed = PhysicalField.FromFunc(
            domain_2D_mixed, u_fn_2d_cheb, name="u_2d_mixed"
        )
        print_verb(
            abs(
                cast(
                    float,
                    cast(
                        PhysicalField, u_2d_mixed.definite_integral(1)
                    ).definite_integral(0),
                )
                - -1.949287106500328240494806919989493133738434465663124816597170852019867576675856194477028450123270962
            ),
            verbosity_level=3,
        )
        self.assertTrue(
            (
                abs(
                    cast(
                        float,
                        cast(
                            PhysicalField, u_2d_mixed.definite_integral(1)
                        ).definite_integral(0),
                    )
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
        domain_2D_mixed_2 = PhysicalDomain.create(
            (Nx, Ny), (True, False), scale_factors=(sc_x, 1.0)
        )
        u_2d_mixed_2 = PhysicalField.FromFunc(
            domain_2D_mixed_2, u_fn_2d_cheb, name="u_2d_mixed_2"
        )
        print_verb(
            abs(
                cast(
                    float,
                    cast(
                        PhysicalField, u_2d_mixed_2.definite_integral(1)
                    ).definite_integral(0),
                )
                - -0.9746435532501641202474034599947465668692172328315624082985854260099337883379280972385142250616354812
            ),
            verbosity_level=3,
        )
        self.assertTrue(
            (
                abs(
                    cast(
                        float,
                        cast(
                            PhysicalField, u_2d_mixed_2.definite_integral(1)
                        ).definite_integral(0),
                    )
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
        u_3d_fourier_int = cast(
            float,
            cast(
                PhysicalField,
                cast(
                    PhysicalField, u_3d_fourier.definite_integral(2)
                ).definite_integral(1),
            ).definite_integral(0),
        )
        print_verb(u_3d_fourier_int, verbosity_level=3)
        self.assertTrue((abs(u_3d_fourier_int - -10.84981433261992)) < tol)
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
        u_3d_cheb = PhysicalField.FromFunc(
            domain_3D_cheb, u_fn_3d_cheb, name="u_3d_cheb"
        )
        u_3d_cheb_int = cast(
            float,
            cast(
                PhysicalField,
                cast(PhysicalField, u_3d_cheb.definite_integral(2)).definite_integral(
                    1
                ),
            ).definite_integral(0),
        )
        print_verb(u_3d_cheb_int, verbosity_level=3)
        self.assertTrue((abs(u_3d_cheb_int - 10.128527022082872)) < tol)
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
        u_3d_mixed = PhysicalField.FromFunc(
            domain_3D_mixed, u_fn_3d_mixed, name="u_3d_mixed"
        )
        u_3d_mixed_int = cast(
            float,
            cast(
                PhysicalField,
                cast(PhysicalField, u_3d_mixed.definite_integral(2)).definite_integral(
                    1
                ),
            ).definite_integral(0),
        )
        print_verb(u_3d_mixed_int, verbosity_level=3)
        self.assertTrue((abs(u_3d_mixed_int - 7.596395266449558)) < tol)
        self.assertTrue((abs(u_3d_mixed.volume_integral() - 7.596395266449558)) < tol)

    def test_poisson_slices(self) -> None:
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
        domain_y = PhysicalDomain.create((Ny,), (False,)).hat()

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

        def solve_poisson_for_single_wavenumber(kx: int, kz: int) -> jnp_array:
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
        out = out_hat.no_hat()

        # u_ana.plot(out)

        tol = 1e-8
        print_verb(abs(u_ana - out), verbosity_level=3)
        self.assertTrue(abs(u_ana - out) < tol)

    def test_poisson_no_slices(self) -> None:
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
        print_verb(abs(u_ana - out), verbosity_level=3)
        self.assertTrue(abs(u_ana - out) < tol)

    def test_navier_stokes_laminar(
        self, Ny: int = 96, perturbation_factor: float = 0.01
    ) -> None:
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

            def before_time_step(nse_: Equation) -> None:
                nse = cast(NavierStokesVelVort, nse_)
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
            vel_x_ana = PhysicalField.FromFunc(
                nse.get_physical_domain(), vel_x_fn_ana, name="vel_x_ana"
            )

            print_verb("Doing post-processing")
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            tol = 6e-5
            print_verb(abs(vel[0] - vel_x_ana), verbosity_level=2)
            print_verb(abs(vel[1]), verbosity_level=2)
            print_verb(abs(vel[2]), verbosity_level=2)
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
    #     print_verb(errors)

    #     def fittingFunc(x, a, b):
    #         return a + b * x

    #     result = optimization.curve_fit(fittingFunc, Nys, errorsLog)
    #     print_verb(result)

    def test_linear_stability(self) -> None:
        n = 50
        # n = 4
        Re = 5772.22
        alpha = 1.02056

        lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=n)
        evs, _ = lsc.calculate_eigenvalues()
        # evs, evecs = lsc.calculate_eigenvalues()
        print_verb(evs[0], verbosity_level=3)
        # print_verb(evecs[0])
        self.assertTrue(evs[0].real <= 0.0 and evs[0].real >= -1e-8)

    def test_perturbation_laminar(
        self, Ny: int = 48, perturbation_factor: float = 0.01
    ) -> None:
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

            print_verb("Doing post-processing")
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            tol = 5e-7
            print_verb(abs(vel[0]), verbosity_level=3)
            print_verb(abs(vel[1]), verbosity_level=3)
            print_verb(abs(vel[2]), verbosity_level=3)
            # check that the simulation is really converged
            self.assertTrue(abs(vel[0]) < tol)
            self.assertTrue(abs(vel[1]) < tol)
            self.assertTrue(abs(vel[2]) < tol)

    # def test_2d_growth(self):
    #     growth_5500_data = run_pseudo_2d_perturbation(Re=5500, end_time=1.5, eps=1e-1, linearize=True, Nx=4, Ny=90, Nz=2, plot=True, jit=False)
    #     growth_6000_data = run_pseudo_2d_perturbation(Re=6000, end_time=1.5, eps=1e-1, linearize=True, Nx=4, Ny=90, Nz=2, plot=True, jit=False)
    #     growth_5500 = []
    #     growth_6000 = []
    #     for i in range(3):
    #         growth_5500.append(growth_5500_data[i][-1] - growth_5500_data[i][-2])
    #         growth_6000.append(growth_6000_data[i][-1] - growth_6000_data[i][-2])
    #     print_verb("growth_5500: ", growth_5500)
    #     print_verb("growth_6000: ", growth_6000)
    #     self.assertTrue(
    #         all([growth < 0 for growth in growth_5500]),
    #         "Expected perturbations to decay for Re=5500.",
    #     )
    #     self.assertTrue(
    #         all([growth > 0 for growth in growth_6000]),
    #         "Expected perturbations to increase for Re=6000.",
    #     )
    #     vel_final_jit_5500 = growth_5500_data[-1]
    #     vel_final_jit_6000 = growth_6000_data[-1]

    #     # Now check that the same result is obtained when using solve_scan. To
    #     # simplify and strengthen the test, only compare the final velocity
    #     # fields.
    #     data_no_jit_5500 = run_pseudo_2d_perturbation(Re=5500, end_time=1.5, eps=1e-1, linearize=True, Nx=4, Ny=90, Nz=2)
    #     data_no_jit_6000 = run_pseudo_2d_perturbation(Re=6000, end_time=1.5, eps=1e-1, linearize=True, Nx=4, Ny=90, Nz=2)
    #     vel_final_no_jit_5500 = data_no_jit_5500[-1]
    #     vel_final_no_jit_6000 = data_no_jit_6000[-1]

    #     print_verb((vel_final_jit_5500 - vel_final_no_jit_5500).energy())
    #     print_verb((vel_final_jit_6000 - vel_final_no_jit_6000).energy())
    #     self.assertTrue(
    #         (vel_final_jit_5500 - vel_final_no_jit_5500).energy() < 1e-6
    #     )
    #     self.assertTrue(
    #         (vel_final_jit_6000 - vel_final_no_jit_6000).energy() < 1e-6
    #     )

    def test_2d_growth_rates_quantitatively(self) -> None:
        def run_re(
            Re: float, rotated: bool = False, use_antialiasing: bool = True
        ) -> "pseudo_2d_perturbation_return_type":
            end_time = 6e-1
            if use_antialiasing:
                N = 4
                aliasing = 3 / 2
            else:
                N = 6
                aliasing = 1
            if rotated:
                return run_pseudo_2d_perturbation(
                    Re=Re,
                    end_time=end_time,
                    Nx=N,
                    Ny=64,
                    Nz=N,
                    linearize=True,
                    plot=True,
                    save=False,
                    eps=1.0,
                    dt=1e-2,
                    rotated=True,
                    aliasing=aliasing,
                )
            else:
                return run_pseudo_2d_perturbation(
                    Re=Re,
                    end_time=end_time,
                    Nx=N,
                    Ny=64,
                    Nz=N,
                    linearize=True,
                    plot=True,
                    save=False,
                    eps=1.0,
                    dt=1e-2,
                    aliasing=aliasing,
                )

        def run(
            rotated: bool = False, use_antialiasing: bool = False
        ) -> tuple[list[list[float]], jnp_array, jnp_array]:
            ts = []
            energy = []
            energy_ana = []
            for Re in [5500, 5772.22, 6000]:
                out = run_re(Re, rotated, use_antialiasing)
                ts.append(out[-2])
                energy.append(out[0])
                energy_ana.append(out[3])
            return (ts, jnp.array(energy), jnp.array(energy_ana))

        def calculate_growth_rates(
            ts: list[list[float]], energy: jnp_array, energy_ana: jnp_array
        ) -> None:
            start_index = 1  # don't start at 0 to allow for some initial transient
            time = ts[0][-1] - ts[0][start_index]
            print_verb("Re = 5500:")
            growth_rate = (energy[0][-1] - energy[0][start_index]) / (
                time * energy_ana[0][start_index]
            )
            growth_rate_ana = (energy_ana[0][-1] - energy_ana[0][start_index]) / (
                time * energy_ana[0][start_index]
            )
            print_verb("growth rate: ", growth_rate)
            print_verb("growth rate (analytical): ", growth_rate_ana)
            print_verb("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_5500 = abs(
                (growth_rate_ana - growth_rate)
                / (0.5 * (growth_rate_ana + growth_rate))
            )
            print_verb("relative error: ", rel_error_5500)
            assert rel_error_5500 < 1e-3

            print_verb("Re = 5772.22:")
            growth_rate = (energy[1][-1] - energy[1][start_index]) / (
                time * energy_ana[1][start_index]
            )
            growth_rate_ana = (energy_ana[1][-1] - energy_ana[1][start_index]) / (
                time * energy_ana[1][start_index]
            )
            print_verb("growth rate: ", growth_rate)
            print_verb("growth rate (analytical): ", growth_rate_ana)
            print_verb("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_5772 = abs(
                (growth_rate_ana - growth_rate)
                / (0.5 * (growth_rate_ana + growth_rate))
            )
            print_verb("relative error: ", rel_error_5772)
            assert (
                rel_error_5772 < 2
            )  # (this can be quite large as the denominator is almost zero)
            assert abs(growth_rate) < 2e-6

            print_verb("Re = 6000:")
            growth_rate = (energy[2][-1] - energy[2][start_index]) / (
                time * energy_ana[2][start_index]
            )
            growth_rate_ana = (energy_ana[2][-1] - energy_ana[2][start_index]) / (
                time * energy_ana[2][start_index]
            )
            print_verb("growth rate: ", growth_rate)
            print_verb("growth rate (analytical): ", growth_rate_ana)
            print_verb("difference: ", abs(growth_rate_ana - growth_rate))
            rel_error_6000 = abs(
                (growth_rate_ana - growth_rate)
                / (0.5 * (growth_rate_ana + growth_rate))
            )
            print_verb("relative error: ", rel_error_6000)
            assert rel_error_6000 < 1e-3

        def plot(
            ts: list[list[float]],
            dataset: jnp_array,
            dataset_ana: jnp_array,
            rotated: bool,
            use_antialiasing: bool,
        ) -> None:
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ax.plot(
                ts[0],
                dataset_ana[0] / dataset_ana[0][0],
                "b-",
                label="Re_5500 (linear theory)",
            )
            ax.plot(ts[0], dataset[0] / dataset_ana[0][0], "bx", label="Re_5500 (DNS)")
            ax.plot(
                ts[1],
                dataset_ana[1] / dataset_ana[1][0],
                "y",
                label="Re_5772 (linear theory)",
            )
            ax.plot(ts[1], dataset[1] / dataset_ana[1][0], "kx", label="Re_5772 (DNS)")
            ax.plot(
                ts[2],
                dataset_ana[2] / dataset_ana[2][0],
                "k-",
                label="Re_6000 (linear theory)",
            )
            ax.plot(ts[2], dataset[2] / dataset_ana[2][0], "gx", label="Re_6000 (DNS)")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$G$")
            fig.legend(loc="upper left")
            fig.savefig(
                "plots/"
                + "energy_"
                + ("rotated" if rotated else "normal")
                + "_domain_"
                + ("with" if use_antialiasing else "without")
                + "_antialiasing"
                + ".png"
            )

        def main() -> None:
            for use_antialiasing in [True, False]:
                for rotated in [False, True]:
                    print_verb(
                        "testing growth rates in "
                        + ("rotated" if rotated else "normal")
                        + " domain "
                        + ("with" if use_antialiasing else "without")
                        + " antialiasing"
                    )
                    ts, energy, energy_ana = run(rotated, use_antialiasing)
                    plot(ts, energy, energy_ana, rotated, use_antialiasing)
                    calculate_growth_rates(ts, energy, energy_ana)

        main()

    def test_transient_growth(self) -> None:
        for Re, t, err in [(600, 2, 8e-4), (3000, 15, 2e-5)]:
            gain, expected_gain, _, _ = run_transient_growth(
                Re, t, alpha=1, beta=0, plot=False
            )
            rel_error = abs((gain - expected_gain) / expected_gain)
            print_verb(rel_error, verbosity_level=3)
            assert rel_error < err
        for Re, t, err in [(600, 2, 5e-6), (3000, 15, 5e-2)]:
            gain, expected_gain, _, _ = run_transient_growth_nonpert(
                Re, t, alpha=1, beta=0, eps=1e-3, plot=False
            )
            rel_error = abs((gain - expected_gain) / expected_gain)
            print_verb(rel_error, verbosity_level=3)
            assert rel_error < err


if __name__ == "__main__":
    unittest.main()

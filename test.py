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
import scipy.optimize as optimization
import time

import numpy as np

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
    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
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
    # print(abs(u_x - u_x_ana))
    # print(abs(u_xx - u_xx_ana))
    assert abs(u_x - u_x_ana) < tol
    assert abs(u_xx - u_xx_ana) < tol


def test_2D():
    Nx = 20
    # Ny = Nx
    Ny = 24
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
    Nx = 24
    Ny = 40
    Nz = 20
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

    u_hat_int = u_hat.integrate(0)
    u_int = u_hat_int.no_hat()
    u_int.name = "u_1d_int"
    u_hat_int_2 = u_hat.integrate(0, 2)
    u_int_2 = u_hat_int_2.no_hat()
    u_int_2.name = "u_1d_int_2"
    tol = 1e-6

    u_hat_diff = u_hat.diff(0, 1)
    u_hat_diff_2 = u_hat.diff(0, 2)
    u_diff = u_hat_diff.no_hat()
    u_diff.name = "u_1d_diff"
    u_diff_2 = u_hat_diff_2.no_hat()
    u_diff_2.name = "u_1d_diff_2"
    u.plot(u_diff, u_diff_2, u_int, u_int_2)
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
    domain = Domain((Nx, Ny), (True, True))

    u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[1])
    u = Field.FromFunc(domain, func=u_fn, name="u_2d")

    u_x_fn = lambda X: -jnp.sin(X[0]) * jnp.cos(X[1])
    u_x_ana = Field.FromFunc(domain, func=u_x_fn, name="u_2d_x_ana")
    u_xx_fn = lambda X: -jnp.cos(X[0]) * jnp.cos(X[1])
    u_xx_ana = Field.FromFunc(domain, func=u_xx_fn, name="u_2d_xx_ana")

    u_y_fn = lambda X: -jnp.cos(X[0]) * jnp.sin(X[1])
    u_y_ana = Field.FromFunc(domain, func=u_y_fn, name="u_2d_y_ana")
    u_yy_fn = lambda X: -jnp.cos(X[0]) * jnp.cos(X[1])
    u_yy_ana = Field.FromFunc(domain, func=u_yy_fn, name="u_2d_yy_ana")

    u_int_x_fn = lambda X: jnp.sin(X[0]) * jnp.cos(X[1])
    u_int_x_ana = Field.FromFunc(domain, func=u_int_x_fn, name="u_2d_int_x_ana")
    u_int_xx_fn = lambda X: -jnp.cos(X[0]) * jnp.cos(X[1])
    u_int_xx_ana = Field.FromFunc(domain, func=u_int_xx_fn, name="u_2d_int_xx_ana")

    u_int_y_fn = lambda X: jnp.cos(X[0]) * jnp.sin(X[1])
    u_int_y_ana = Field.FromFunc(domain, func=u_int_y_fn, name="u_2d_int_y_ana")
    u_int_yy_fn = lambda X: -jnp.cos(X[0]) * jnp.cos(X[1])
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
    u_x.plot()
    u_y.plot()
    u.plot_center(0, u_x, u_x_2, u_int_x, u_int_xx)
    u.plot_center(1, u_y, u_y_2, u_int_y, u_int_yy)
    tol = 2e-7
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
    Nz = Nx - 4
    domain = Domain((Nx, Ny, Nz), (True, False, True))

    # u_fn = lambda X: jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi/2)
    u_0_fn = lambda X: 0.0 * jnp.cos(X[0]) * jnp.cos(X[2]) * jnp.cos(X[1] * jnp.pi / 2)
    u_fn = lambda X: 0.0 * jnp.cos(X[0]) * jnp.cos(X[2]) + jnp.sin(X[1] * jnp.pi)
    u = Field.FromFunc(domain, func=u_fn, name="u_3d")
    v = Field.FromFunc(domain, func=u_0_fn, name="v_3d")
    w = Field.FromFunc(domain, func=u_0_fn, name="w_3d")
    U = VectorField([u, v, w])
    U_hat = U.hat()
    U_nohat = U_hat.no_hat()
    U_nohat.plot(U, U_hat)
    # U.plot(U_hat)
    # U_nohat.plot(U)


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
    u_int_1.plot(u, u_int_1_ana)
    u_int_2.plot(u, u_int_2_ana)

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
    u_int.plot(u_int_ana)

    tol = 1e-8
    # print(abs(u_int - u_int_ana))
    assert abs(u_int - u_int_ana) < tol


def test_poisson_slices():
    Nx = 24
    Ny = Nx + 4
    Nz = Nx - 4

    domain = Domain((Nx, Ny, Nz), (True, False, True))
    domain_y = Domain((Ny,), (False))

    rhs_fn = (
        lambda X: -(2 + jnp.pi**2 / 4)
        * jnp.sin(X[0])
        * jnp.sin(X[2] + 1.0)
        * jnp.cos(X[1] * jnp.pi / 2)
    )
    rhs = Field.FromFunc(domain, rhs_fn, name="rhs")

    u_ana_fn = (
        lambda X: jnp.sin(X[0]) * jnp.sin(X[2] + 1.0) * jnp.cos(X[1] * jnp.pi / 2)
    )
    u_ana = Field.FromFunc(domain, u_ana_fn, name="u_ana")
    rhs_hat = rhs.hat()
    rhs_nohat = rhs_hat.no_hat()

    def solve_poisson_for_single_wavenumber(kx_, kz_):
        kx, kz = int(kx_), int(kz_)
        if kx == 0 or kz == 0:
            # assumes homogeneneous Dirichlet boundary conditions
            return FourierFieldSlice(
                domain_y, 1, rhs_hat[kx, :, kz] * 0.0, "rhs_t_slice", kx, kz
            )
        rhs_hat_slice = FourierFieldSlice(
            domain_y, 1, rhs_hat[kx, :, kz], "rhs_hat_slice", kx, kz
        )
        out = rhs_hat_slice.solve_poisson()
        return out

    out_hat = rhs_hat.reconstruct_from_wavenumbers(solve_poisson_for_single_wavenumber)
    out = out_hat.no_hat()

    u_ana.plot(out)

    tol = 1e-8
    print(abs(u_ana - out))
    assert abs(u_ana - out) < tol


def test_navier_stokes_laminar(Ny=40, pertubation_factor=0.1):

    Re = 1e0

    end_time = 8
    nse = solve_navier_stokes_laminar(Re=Re, Ny=Ny, end_time=end_time, pertubation_factor=pertubation_factor)
    nse.solve()

    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = Field.FromFunc(nse.domain, vel_x_fn_ana, name="vel_x_ana")

    vel_0 = nse.get_initial_field("velocity_hat").no_hat()
    print("Doing post-processing")
    for i in range(nse.time_step)[-4:]:
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        vel[0].plot_center(1, vel_0[0], vel_x_ana)
        vel[1].plot_center(1, vel_0[1])
        vel[2].plot_center(1, vel_0[2])
        tol = 6e-5
        print(abs(vel[0] - vel_x_ana))
        print(abs(vel[1]))
        print(abs(vel[2]))
        # check that the simulation is really converged
        assert abs(vel[0] - vel_x_ana) < tol
        assert abs(vel[1]) < tol
        assert abs(vel[2]) < tol


def test_navier_stokes_laminar_convergence():
    Nys = [24, 48, 96]
    end_time = 10
    def run(Ny):
        nse = solve_navier_stokes_laminar(Re=1, end_time=end_time, Ny=Ny, pertubation_factor=0)
        nse.solve()
        vel_x_fn_ana = lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
        # vel_x_ana = Field.FromFunc(nse.domain, vel_x_fn_ana, name="vel_x_ana")
        vel_hat = nse.get_latest_field("velocity_hat")
        vel = vel_hat.no_hat()
        return vel[0].l2error(vel_x_fn_ana)
    errors = list(map(run, Nys))
    errorsLog = list(map(lambda x: np.log2(x), errors))
    print(errors)

    def fittingFunc(x, a, b):
        return a + b*x
    result = optimization.curve_fit(fittingFunc, Nys, errorsLog)
    print(result)

def test_optimization():
    Re = 1e0
    Ny = 24
    end_time = 1

    nse = solve_navier_stokes_laminar(Re=Re, Ny=Ny, end_time=end_time, pertubation_factor=0.0)
    def run(v0):
        nse_ = solve_navier_stokes_laminar(Re=Re, Ny=Ny, end_time=end_time, max_iter=10, pertubation_factor=0.0)
        nse_.max_iter = 10
        v0_field = Field(nse_.domain, v0)
        vel_0 = nse_.get_initial_field("velocity_hat")
        vel_0_new = VectorField([v0_field.hat(), vel_0[1], vel_0[2]])
        nse_.set_field("velocity_hat", 0, vel_0_new)
        nse_.after_time_step_fn = None
        nse_.solve()
        vel_out = nse_.get_latest_field("velocity_hat").no_hat()
        return (vel_out[0].max() / vel_0[0].max()).real

    vel_x_fn_ana = lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    v0_0 = Field.FromFunc(nse.domain, vel_x_fn_ana)

    v0s = [v0_0.field]
    eps = 1e3
    for i in range(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_field = Field(nse.domain, corr, name="correction")
        corr_field.update_boundary_conditions()
        print("gain: " + str(gain))
        print("corr (abs): " + str(abs(corr_field)))
        v0s.append(v0s[-1] + eps * corr_field.field)
        v0_new = Field(nse.domain, v0s[-1])
        v0_new.name = "vel_0_" + str(i)
        v0_new.plot(v0_0)

def test_navier_stokes_turbulent():

    Re = 1.8e2

    end_time = 100
    # nse = solve_navier_stokes_laminar(Re=Re, Ny=96, Nx=48, end_time=end_time, pertubation_factor=1)
    nse = solve_navier_stokes_laminar(Re=Re, Ny=12, Nx=4, end_time=end_time, pertubation_factor=1)

    vel_0 = nse.get_initial_field("velocity_hat").no_hat()
    def after_time_step(nse):
        i = nse.time_step
        if i > 1:
            vel = nse.get_field("velocity_hat", i).no_hat()
            vel[0].plot()
            vel[0].plot_center(1, vel_0[0])
            vel[1].plot_center(1, vel_0[1])
            vel[2].plot_center(1, vel_0[2])
    nse.after_time_step_fn = after_time_step
    nse.solve()

def test_vmap():
    def fn(x):
        return jax.lax.cond(x > 3, lambda x_: x_*2, lambda x_: x_*3, x)
        # if x > 3:
        #     return x*2
        # else:
        #     return x*3
    xs = jnp.arange(10)
    out = jax.vmap(fn)(xs)
    print(xs)
    print(out)


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
    test_poisson_slices()
    # test_navier_stokes_laminar()
    # test_navier_stokes_laminar_convergence()
    # test_optimization()
    # test_navier_stokes_turbulent()
    # test_vmap()

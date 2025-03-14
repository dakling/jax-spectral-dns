#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.main import get_args_from_yaml_file

from jax.scipy.optimize import minimize

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import glob

from typing import List, TypeVar
import pyvista as pv
import numpy as np
import h5py
import matplotlib
from matplotlib import figure
from matplotlib.axes import Axes
import jax.numpy as jnp
from jax_spectral_dns.cheb import cheby
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, VectorField, PhysicalField, FourierField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

matplotlib.set_loglevel("error")

from PIL import Image

# matplotlib.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']
font_size = 18
matplotlib.rcParams.update(
    {
        "font.size": font_size,
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)
pv.global_theme.font.size = font_size
pv.global_theme.font.title_size = font_size
pv.global_theme.font.label_size = font_size


# STORE_PREFIX = "/store/DAMTP/dsk34"
STORE_PREFIX = "/data/septal/dsk34"
HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)


def get_domain(shape, Lx_over_pi: float, Lz_over_pi: float):
    return PhysicalDomain.create(
        shape,
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
    )


def get_vel_field_minimal_channel(domain: PhysicalDomain):

    cheb_coeffs = np.loadtxt(
        HOME_DIR_BASE + "/profiles/Re_tau_180_90_small_channel.csv", dtype=np.float64
    )

    Ny = domain.number_of_cells(1)
    U_mat = np.zeros((Ny, len(cheb_coeffs)))
    for i in range(Ny):
        for j in range(len(cheb_coeffs)):
            U_mat[i, j] = cheby(j, 0)(domain.grid[1][i])
    U_y_slice = U_mat @ cheb_coeffs

    slice_domain = PhysicalDomain.create(
        (domain.get_shape_aliasing()[1],),
        (False,),
        scale_factors=(1.0,),
        aliasing=1,
    )
    U_y_slice_field = average_y_symm(
        PhysicalField(slice_domain, jnp.array(U_y_slice), time_step=0)
    )

    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data = jnp.moveaxis(
        jnp.tile(jnp.tile(U_y_slice_field.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)),
        1,
        2,
    )
    vel_base = VectorField(
        [
            PhysicalField(domain, jnp.asarray(u_data)),
            PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
        ]
    )
    vel_base.set_name("average_velocity_vilda")
    return vel_base


# def average_y_symm(vel: VectorField[PhysicalField]) -> VectorField[PhysicalField]:
def average_y_symm(vel: PhysicalField) -> PhysicalField:
    y_axis = 0  # assuming vel is only a slice
    # vel_flip = VectorField(
    #     [
    #         PhysicalField(
    #             vel.get_physical_domain(), jnp.flip(v.get_data(), axis=y_axis)
    #         )
    #         for v in vel
    #     ]
    # )
    vel_flip = PhysicalField(
        vel.get_physical_domain(), jnp.flip(vel.get_data(), axis=y_axis)
    )
    out = 0.5 * (vel_flip + vel)
    out.set_name(vel.get_name() + "_y_symm")
    return out


REPORTED = False


def get_base_perturbation(
    domain: PhysicalDomain, mean_perturbation: float, *b: float
) -> "VectorField[PhysicalField]":

    try:
        slice_domain = PhysicalDomain.create(
            (domain.get_shape_aliasing()[1],),
            (False,),
            scale_factors=(1.0,),
            aliasing=1,
        )
        vel_base_turb_slice = average_y_symm(
            PhysicalField.FromFile(
                slice_domain, "average_velocity", name="average_velocity_x"
            )
        )
        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = jnp.moveaxis(
            jnp.tile(
                jnp.tile(vel_base_turb_slice.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)
            ),
            1,
            2,
        )
        vel_avg = VectorField(
            [
                PhysicalField(domain, jnp.asarray(u_data)),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            ]
        )
        vel_avg.set_name("vel_avg")
    except FileNotFoundError:
        global REPORTED
        if not REPORTED:
            print("using vilda's profile")
        REPORTED = True
        vel_avg = get_vel_field_minimal_channel(domain)
    return vel_avg + VectorField(
        [
            PhysicalField.FromFunc(
                domain,
                lambda X: mean_perturbation
                * (
                    # (1 - b[0] + b[1] - b[2]) # TODO improve this
                    (1 - b[0] + b[1])  # TODO improve this
                    + jnp.cos(jnp.pi * X[1])
                    + b[0] * jnp.cos(2 * jnp.pi * X[1])
                    + b[1] * jnp.cos(3 * jnp.pi * X[1])
                    # + b[2] * jnp.cos(4 * jnp.pi * X[1])
                )
                + 0.0 * X[2],
            ),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
        ]
    )


def post_process_histograms() -> None:

    Lx_over_pi = 0.6
    Lz_over_pi = 0.3
    Nx, Ny, Nz = (48, 129, 60)
    domain = get_domain((Nx, Ny, Nz), Lx_over_pi, Lz_over_pi)
    slice_domain = PhysicalDomain.create(
        (domain.get_shape_aliasing()[1],),
        (False,),
        scale_factors=(1.0,),
        aliasing=1,
    )
    filenames = sorted(
        glob.glob("vel_hist_bin_*", root_dir=Field.field_dir),
        key=lambda f: os.path.getmtime("fields/" + f),
    )

    vel_avg = get_vel_field_minimal_channel(domain)
    fig = figure.Figure(figsize=(15, 12))
    # ax = fig.subplots(3, 3)
    ax = fig.subplots(3, 6)

    for fl in filenames:
        i = fl.split("_")[-1]
        vel_base_turb_slice = PhysicalField.FromFile(
            slice_domain, fl, "hist_bin_" + i + "_x", time_step=0
        )
        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = np.moveaxis(
            np.tile(
                np.tile(vel_base_turb_slice.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)
            ),
            1,
            2,
        )
        vel_base_turb = VectorField(
            [
                PhysicalField(domain, jnp.asarray(u_data)),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            ]
        )
        vel_base_turb.set_name("vel_avg_" + i)
        j = int(i) // 3
        k = int(i) % 3
        vel_base_turb[0].plot_center(1, vel_avg[0], ax=ax[k][j], fig=fig)
    fig.savefig("plots/hist.png")

    # def fn(A) -> float:
    #     return (vel_base_turb[0] - get_base_perturbation(domain, *A)[0]).energy()

    # result = minimize(
    #     fn, x0=jnp.array([0.0,-1.0, -1.0]),
    #     method='BFGS')
    # # j = 0
    # # a = 1.0
    # # b = 1.0
    # # max_iter = 100
    # # tol = 1e-25
    # # while abs(fn(jnp.array([a,b]))) > tol and j < max_iter:
    # #     A = jnp.array([a, b])
    # #     gr = jax.grad(fn)(A)
    # #     hess = jax.hessian(fn)(A)
    # #     step = - jnp.linalg.inv(hess) @ gr
    # #     a += step[0]
    # #     b += step[1]
    # #     j += 1

    # print_verb("histogram:", i)
    # print_verb("a:", result.x[0])
    # print_verb("bs:", result.x[1:])
    # # print_verb("c:", c)
    # print_verb("err:", abs(fn(result.x)))
    # fitted_profile = get_base_perturbation(domain, *result.x)
    # fitted_profile.set_name("fit")
    # vel_base_turb[0].plot_center(1, fitted_profile[0])


post_process_histograms()

#!/usr/bin/env python3


import os
import sys

from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import glob

from typing import Iterable, List, TypeVar, Tuple
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

args = get_args_from_yaml_file(HOME_DIR_BASE + "/simulation_settings.yml")
print(args)


def get_domain():
    Lx_over_pi = args.get("Lx_over_pi", 2.0)
    Lz_over_pi = args.get("Lz_over_pi", 1.0)
    Nx = args.get("Nx", 64)
    Ny = args.get("Ny", 129)
    Nz = args.get("Nz", 80)
    return PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
        aliasing=3 / 2,
        dealias_nonperiodic=False,
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
    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data = np.moveaxis(
        np.tile(np.tile(U_y_slice, reps=(nz, 1)), reps=(nx, 1, 1)), 1, 2
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


def plot():
    domain = get_domain()

    slice_domain = PhysicalDomain.create(
        (domain.get_shape_aliasing()[1],),
        (False,),
        scale_factors=(1.0,),
        aliasing=1,
    )
    avg_slice = PhysicalField.FromFile(
        slice_domain, "average_velocity", name="average_velocity_x"
    )
    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data = np.moveaxis(
        np.tile(np.tile(avg_slice.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)),
        1,
        2,
    )
    avg_ = PhysicalField(domain, jnp.asarray(u_data))

    avg_slice_old = PhysicalField.FromFile(
        slice_domain, "average_velocity_old", name="average_velocity_x"
    )
    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data_old = np.moveaxis(
        np.tile(np.tile(avg_slice_old.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)),
        1,
        2,
    )
    avg_old = PhysicalField(domain, jnp.asarray(u_data_old))

    vilda = get_vel_field_minimal_channel(domain)

    avg_.plot_center(1, avg_old, vilda[0])
    print((avg_ - avg_old).energy())


plot()

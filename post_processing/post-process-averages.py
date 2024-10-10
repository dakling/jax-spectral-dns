#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import glob

from typing import List
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


def post_process_averages() -> None:

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
    # avg_vels = []
    # for f in glob.glob("avg_vel_20*", root_dir="./fields"):
    #     avg_vels.append(VectorField.FromFile(domain, f, "average_velocity"))
    vel_00_s = []
    for fl in glob.glob("trajectory_00_20*", root_dir="./fields"):
        with h5py.File(fl, "r") as f:
            velocity_00_trajectory = f["trajectory_00"]
            n_steps = velocity_00_trajectory.shape[0]
            for i in range(n_steps):
                vel_00_s.append(
                    VectorField.FromData(
                        PhysicalField,
                        domain,
                        velocity_00_trajectory[i],
                        name="velocity_00",
                    )
                )

    def avg_fields(fs: List[VectorField[PhysicalField]]) -> VectorField[PhysicalField]:
        out = fs[0] * 0.0
        for f in fs:
            out += f
        return out / len(fs)

    # print_verb("Taking the average of", len(avg_vels), "snapshots (equally weighted!)")
    print_verb("Taking the average of", len(vel_00_s), "snapshots (equally weighted!)")
    # avg = avg_fields(avg_vels)
    avg = avg_fields(vel_00_s)
    print_verb("average y-velocity range: [", avg[1].max(), ",", avg[1].min(), "]")
    print_verb("average z-velocity range: [", avg[2].max(), ",", avg[2].min(), "]")
    avg.set_name("average_velocity_single")
    avg[0].plot_center(0)
    avg.set_name("average_velocity_ensemble")
    # avg[0].plot_center(1, *[avg_vel[0] for avg_vel in avg_vels])
    avg[0].plot_center(0, *[avg_vel[0] for avg_vel in vel_00_s])
    avg.set_name("average_velocity")
    avg[0].plot_center(0, get_vel_field_minimal_channel(domain)[0])
    # avg.plot_3d(0)
    # avg.plot_3d(1)
    # avg.plot_3d(2)
    mass_flux = []
    vel_cl = []
    for f in vel_00_s:
        mass_flux.append(f[0].volume_integral())
        vel_cl.append(f[0][Ny // 2])
    fig = figure.Figure()
    ax = fig.subplots(2, 1)
    ax[0].plot(mass_flux)
    ax[1].plot(vel_cl)
    fig.savefig("plots/mass_flux_over_time.png")


post_process_averages()

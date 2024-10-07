#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation
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


STORE_PREFIX = "/store/DAMTP/dsk34"
HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)


def get_domain(shape, Lx_over_pi: float, Lz_over_pi: float):
    return PhysicalDomain.create(
        shape,
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
    )


def post_process_averages() -> None:

    Lx_over_pi = 0.6
    Lz_over_pi = 0.3
    domain = get_domain((48, 129, 60), Lx_over_pi, Lz_over_pi)
    avg_vels = []
    for f in glob.glob("avg_vel_*", root_dir="./fields"):
        avg_vels.append(VectorField.FromFile(domain, f, "average_velocity"))

    def avg_fields(fs: List[VectorField[PhysicalField]]) -> VectorField[PhysicalField]:
        out = fs[0] * 0.0
        for f in fs:
            out += f
        return out / len(fs)

    avg = avg_fields(avg_vels)
    avg[0].plot_center(1, *[avg_vel[0] for avg_vel in avg_vels])


post_process_averages()

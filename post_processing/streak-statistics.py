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

data = np.genfromtxt(Field.field_dir + "/streak_data.txt")
ts = data[:, 0]
lambda_y = data[:, 1]
lambda_z = data[:, 2]
streak_amplitude = data[-2000:, 3]

print("mean streak amplitude:", np.mean(streak_amplitude))
print("streak amplitude std:", np.std(streak_amplitude))

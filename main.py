#!/usr/bin/env python3

import jax
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
import os
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")

jax.config.update("jax_enable_x64", True)

from importlib import reload
import sys

try:
    reload(sys.modules["heat_eq"])
except:
    print("Unable to load")
from heat_eq import solve_heat_eq_2D, solve_heat_eq_3D

try:
    reload(sys.modules["navier_stokes"])
except:
    print("Unable to load")
from navier_stokes import solve_navier_stokes_laminar

try:
    reload(sys.modules["navier_stokes_pertubation"])
except:
    print("Unable to load navier-stokes-pertubation")
from navier_stokes_pertubation import solve_navier_stokes_pertubation

try:
    reload(sys.modules["test"])
except:
    print("Unable to load test")
from test import TestProject


try:
    reload(sys.modules["examples"])
except:
    print("Unable to load examples")
from examples import run_jimenez_1990, run_transient_growth


def init():
    newpaths = ['./fields/', "./plots/"]
    for newpath in newpaths:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    # clean plotting dir
    [f.unlink() for f in Path(newpaths[1]).glob("*.pdf") if f.is_file()]
    [f.unlink() for f in Path(newpaths[1]).glob("*.png") if f.is_file()]
    [f.unlink() for f in Path(newpaths[1]).glob("*.mp4") if f.is_file()]

def main():
    init()

    # run_jimenez_1990()
    run_transient_growth()

    # run tests
    # tp = TestProject()
    # tp.test_1D_cheb()
    # tp.test_definite_integral()
    # tp.test_fourier_simple_3D()



main()

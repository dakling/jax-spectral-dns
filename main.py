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
    reload(sys.modules["test_project"])
except:
    if hasattr(sys, 'ps1'):
        print("Unable to load test_project")
from test_project import TestProject

try:
    reload(sys.modules["examples"])
except:
    if hasattr(sys, 'ps1'):
        print("Unable to load examples")
from examples import *


def main():
    # pass
    # print(jax.device_count())
    # run_jimenez_1990()
    # run_transient_growth()
    # run_pseudo_2d_pertubation(Re=5.5e3, Ny=240)

    # run tests
    tp = TestProject()
    # tp.test_linear_stability()
    # tp.test_definite_integral()
    # tp.test_fourier_simple_3D()
    tp.test_2d_growth()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        globals()[sys.argv[1]]()
    else:
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)

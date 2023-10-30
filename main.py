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
    reload(sys.modules["examples"])
except:
    if hasattr(sys, 'ps1'):
        print("Unable to load examples")
from examples import run_jimenez_1990, run_transient_growth


def main():
    # print(jax.device_count())
    # run_jimenez_1990()
    run_transient_growth()

    # run tests
    # tp = TestProject()
    # tp.test_1D_cheb()
    # tp.test_definite_integral()
    # tp.test_fourier_simple_3D()

if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except (IndexError, KeyError):
        main()

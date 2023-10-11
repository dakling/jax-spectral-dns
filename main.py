#!/usr/bin/env python3

import jax
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
import time

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
    reload(sys.modules["test"])
except:
    print("Unable to load")
from test import run_all_tests, run_all_tests_profiling


def main():
    # optimize_fd()
    # optimize_spectral()
    # optimize_channel()
    # optimize_channel_2d()
    run_all_tests()
    # run_all_tests_profiling()

    # mn()
    # solve_heat_eq_2D()
    # solve_heat_eq_3D()

    # solve_navier_stokes_3D_channel()

    # N = int(64 * 180 * 64)
    # N = int(64 * 90 * 64)
    # print(N)
    # A = np.ones((N, N), dtype=np.float)
    # x = np.ones((N,))
    # start_time = time.time()
    # b = A@x
    # end_time = time.time()
    # print(end_time - start_time)


main()

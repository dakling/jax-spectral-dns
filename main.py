#!/usr/bin/env python3

import jax
import jax.scipy as jsp
import jax.numpy as jnp
import jax_cfd
import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
from jax_cfd.spectral import utils as spectral_utils
from jax_cfd.base.grids import GridVariable
import numpy as np
import seaborn as sns
import xarray
import time

from importlib import reload
import sys

try:
    reload(sys.modules["fd"])
except:
    pass
from fd import optimize_fd

try:
    reload(sys.modules["spectral"])
except:
    pass
from spectral import optimize_spectral

try:
    reload(sys.modules["channel"])
except:
    pass
from channel import optimize_channel

try:
    reload(sys.modules["channel2d"])
except:
    pass
from channel2d import optimize_channel_2d

try:
    reload(sys.modules["spectral_channel_solver"])
except:
    pass
from spectral_channel_solver import main as mn

try:
    reload(sys.modules["heat_eq"])
except:
    pass
from heat_eq import perform_simulation_cheb_fourier_2D_no_mat

def main():
    # optimize_fd()
    # optimize_spectral()
    # optimize_channel()
    # optimize_channel_2d()

    # mn()
    perform_simulation_cheb_fourier_2D_no_mat()

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


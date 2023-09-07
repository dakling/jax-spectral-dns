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

from importlib import reload
import sys

try:
    reload(sys.modules["spectral"])
except:
    pass
from spectral import read_state, plot_state, create_grid

def plot():
    grid = create_grid()

    plot_interval = 5
    u0_vec = read_state()
    for i in range(len(u0_vec)):
        if i % plot_interval == 0:
            plot_state(u0_vec[i], grid, i)


plot()

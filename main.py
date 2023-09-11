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

def main():
    # optimize_fd()
    # optimize_spectral()
    # optimize_channel()
    # optimize_channel_2d()
    mn()


main()


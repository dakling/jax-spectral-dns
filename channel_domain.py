#!/usr/bin/env python3

from types import NoneType
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

import numpy as np

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    pass
from domain import Domain

class ChannelDomain(Domain):
    def __init__(self, shape):
        self.number_of_dimensions = len(shape)
        self.periodic_directions = [True, False, True]
        self.grid = []
        self.cheb_mat = self.assemble_cheb_diff_mat(shape[1])
        self.diff_mats = []

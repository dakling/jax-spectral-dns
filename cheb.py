#!/usr/bin/env python3

import numpy as np
from numpy.polynomial.chebyshev import *


def cheb(order, deriv, y=None):
    unit_array = np.eye(order + 1)[order].flatten()
    ch = Chebyshev(unit_array)
    ch = ch.deriv(deriv)
    if y is None:
        return ch
    else:
        return ch(y)


def phi(order, deriv, y):
    return cheb(order + 2, deriv, y) - cheb(order, deriv, y)

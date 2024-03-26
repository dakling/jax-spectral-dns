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


def phi_a(order, deriv, y):
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order + 1, deriv, y) - cheb(2 * order - 1, deriv, y)


def phi_s(order, deriv, y):
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order, deriv, y) - cheb(2 * order - 2, deriv, y)


def phi_as(order, deriv, y, ySym):
    return ySym * phi_a(order, deriv, y) + (1 - ySym) * phi_s(order, deriv, y)


def phi_sa(order, deriv, y, ySym):
    return phi_as(order, deriv, y, 1 - ySym)


def phi_pressure(order, deriv, y):
    order += 1 # compatibility with MATLAB indexing
    return cheb(2*order - 1, deriv, y)

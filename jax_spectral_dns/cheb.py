#!/usr/bin/env python3

from typing import Optional, Union
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from jax_spectral_dns._typing import jsd_float

def cheb(order: int, deriv: int) -> Chebyshev:
    unit_array = np.eye(order + 1)[order].flatten()
    ch: Chebyshev = Chebyshev(unit_array)
    ch: Chebyshev = ch.deriv(deriv) #type: ignore
    return ch


def phi(order: int, deriv: int) -> Chebyshev:
    return cheb(order + 2, deriv) - cheb(order, deriv)


def phi_a(order: int, deriv: int) -> Chebyshev:
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order + 1, deriv) - cheb(2 * order - 1, deriv)


def phi_s(order: int, deriv: int) -> Chebyshev:
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order, deriv) - cheb(2 * order - 2, deriv)


def phi_as(order: int, deriv: int, ySym: jsd_float) -> Chebyshev:
    return ySym * phi_a(order, deriv) + (1 - ySym) * phi_s(order, deriv)


def phi_sa(order: int, deriv: int, ySym: jsd_float) -> Chebyshev:
    return phi_as(order, deriv, 1 - ySym)


def phi_pressure(order: int, deriv: int) -> Chebyshev:
    order += 1 # compatibility with MATLAB indexing
    return cheb(2*order - 1, deriv)

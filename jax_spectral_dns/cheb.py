#!/usr/bin/env python3

from typing import Optional, Union
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def cheb(order: int, deriv: int, y: Optional[float]=None) -> Union[float, Chebyshev]:
    unit_array = np.eye(order + 1)[order].flatten()
    ch = Chebyshev(unit_array)
    ch = ch.deriv(deriv)
    if y is None:
        return ch
    else:
        out: float = ch(y)
        return out


def phi(order: int, deriv: int, y: float) -> Union[float, Chebyshev]:
    return cheb(order + 2, deriv, y) - cheb(order, deriv, y)


def phi_a(order: int, deriv: int, y: float) -> Union[float, Chebyshev]:
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order + 1, deriv, y) - cheb(2 * order - 1, deriv, y)


def phi_s(order: int, deriv: int, y: Optional[float]=None) -> Union[float, Chebyshev]:
    order += 1  # compatibility with MATLAB indexing
    return cheb(2 * order, deriv, y) - cheb(2 * order - 2, deriv, y)


def phi_as(order: int, deriv: int, y: float, ySym: float) -> Union[float, Chebyshev]:
    return ySym * phi_a(order, deriv, y) + (1 - ySym) * phi_s(order, deriv, y)


def phi_sa(order: int, deriv: int, y: float, ySym: float) -> Union[float, Chebyshev]:
    return phi_as(order, deriv, y, 1 - ySym)


def phi_pressure(order: int, deriv: int, y: Optional[float]=None) -> Union[float, Chebyshev]:
    order += 1 # compatibility with MATLAB indexing
    return cheb(2*order - 1, deriv, y)

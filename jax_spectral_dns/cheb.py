#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

if TYPE_CHECKING:
    from jax_spectral_dns._typing import jsd_float


def cheb(order: int, deriv: int) -> Chebyshev:
    unit_array = np.eye(order + 1)[order].flatten()
    ch: Chebyshev = Chebyshev(unit_array)
    ch = ch.deriv(deriv)  # type: ignore[no-untyped-call]
    return ch


def phi(order: int, deriv: int) -> Chebyshev:
    out: Chebyshev = cheb(order + 2, deriv) - cheb(order, deriv)
    return out


def phi_a(order: int, deriv: int) -> Chebyshev:
    order += 1  # compatibility with MATLAB indexing
    out: Chebyshev = cheb(2 * order + 1, deriv) - cheb(2 * order - 1, deriv)
    return out


def phi_s(order: int, deriv: int) -> Chebyshev:
    order += 1  # compatibility with MATLAB indexing
    out: Chebyshev = cheb(2 * order, deriv) - cheb(2 * order - 2, deriv)
    return out


def phi_as(order: int, deriv: int, ySym: "jsd_float") -> Chebyshev:
    out: Chebyshev = ySym * phi_a(order, deriv) + (1 - ySym) * phi_s(order, deriv)
    return out


def phi_sa(order: int, deriv: int, ySym: "jsd_float") -> Chebyshev:
    out: Chebyshev = phi_as(order, deriv, 1 - ySym)
    return out


def phi_pressure(order: int, deriv: int) -> Chebyshev:
    order += 1  # compatibility with MATLAB indexing
    out: Chebyshev = cheb(2 * order - 1, deriv)
    return out

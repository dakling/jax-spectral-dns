#!/usr/bin/env python3
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import dataclasses

from jax_spectral_dns.domain import Domain, PhysicalDomain

if TYPE_CHECKING:
    from jax_spectral_dns._typing import jsd_float, np_complex_array


@dataclasses.dataclass(frozen=True)
class FixedParameters:
    domain: Domain
    dt: "jsd_float"


@dataclasses.dataclass(frozen=True)
class NavierStokesVelVortFixedParameters:
    physical_domain: PhysicalDomain
    poisson_mat: "np_complex_array"
    rk_mats_rhs: "np_complex_array"
    rk_mats_lhs_inv: "np_complex_array"
    rk_rhs_inhom: "np_complex_array"
    rk_mats_lhs_inv_inhom: "np_complex_array"
    rk_mats_rhs_ns: "np_complex_array"
    rk_mats_lhs_inv_ns: "np_complex_array"
    Re_tau: float
    max_cfl: float = 0.3
    dt_update_frequency: int = (
        10  # update the timestep every time_step_udate_frequency time steps
    )
    u_max_over_u_tau: float = 1e0
    number_of_rk_steps: int = 3

#!/usr/bin/env python3

import numpy as np
import dataclasses

from jax_spectral_dns.domain import Domain, PhysicalDomain

@dataclasses.dataclass(frozen=True)
class FixedParameters():
    domain: Domain
    dt: np.float64


@dataclasses.dataclass(frozen=True)
class NavierStokesVelVortFixedParameters():
    physical_domain: PhysicalDomain
    poisson_mat: np.ndarray
    rk_mats_rhs: np.ndarray
    rk_mats_lhs_inv: np.ndarray
    rk_rhs_inhom: np.ndarray
    rk_mats_lhs_inv_inhom: np.ndarray
    rk_mats_rhs_ns: np.ndarray
    rk_mats_lhs_inv_ns: np.ndarray
    Re_tau: np.float64
    max_cfl: np.float64 = 0.3
    dt_update_frequency: int = (
        10  # update the timestep every time_step_udate_frequency time steps
    )
    u_max_over_u_tau: np.float64 = 1e0

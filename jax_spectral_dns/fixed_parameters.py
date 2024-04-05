#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import dataclasses

from jax_spectral_dns.domain import Domain, PhysicalDomain

@dataclasses.dataclass(frozen=True)
class FixedParameters():
    domain: Domain
    dt: np.float64


@dataclasses.dataclass(frozen=True)
class NavierStokesVelVortFixedParameters():
    physical_domain: PhysicalDomain
    poisson_mat: npt.NDArray[np.float64]
    rk_mats_rhs: npt.NDArray[np.float64]
    rk_mats_lhs_inv: npt.NDArray[np.float64]
    rk_rhs_inhom: npt.NDArray[np.float64]
    rk_mats_lhs_inv_inhom: npt.NDArray[np.float64]
    rk_mats_rhs_ns: npt.NDArray[np.float64]
    rk_mats_lhs_inv_ns: npt.NDArray[np.float64]
    Re_tau: float
    max_cfl: float = 0.3
    dt_update_frequency: int = (
        10  # update the timestep every time_step_udate_frequency time steps
    )
    u_max_over_u_tau: float = 1e0

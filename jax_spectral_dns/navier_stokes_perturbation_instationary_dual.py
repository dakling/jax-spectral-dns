#!/usr/bin/env python3


from __future__ import annotations
import gc

from jax_spectral_dns.navier_stokes import (
    get_div_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
    helicity_to_nonlinear_terms,
)
from jax_spectral_dns.navier_stokes_perturbation_instationary import (
    NavierStokesVelVortPerturbationInstationary,
)
from jax_spectral_dns.navier_stokes_perturbation_dual import (
    update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric,
)

NoneType = type(None)
import os
import h5py  # type: ignore
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.figure as figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, cast, List, Union
from typing_extensions import Self
import time

try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    pass

# from importlib import reload
import sys

from jax_spectral_dns.fixed_parameters import NavierStokesVelVortFixedParameters
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
from jax_spectral_dns.navier_stokes_perturbation_dual import (
    NavierStokesVelVortPerturbationDual,
)
from jax_spectral_dns.navier_stokes import (
    get_nabla_vel_1_vel_2,
    get_vel_1_nabla_vel_2,
    helicity_to_nonlinear_terms,
)
from jax_spectral_dns.domain import PhysicalDomain, FourierDomain
from jax_spectral_dns.field import (
    Field,
    PhysicalField,
    VectorField,
    FourierField,
    FourierFieldSlice,
)
from jax_spectral_dns.equation import Equation, print_verb

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        jsd_float,
        jnp_array,
        np_jnp_array,
        Vel_fn_type,
        np_complex_array,
    )


class NavierStokesVelVortPerturbationInstationaryDual(
    NavierStokesVelVortPerturbationDual
):
    name = "Dual Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a time-dependent base flow."
    ...

    def __init__(
        self,
        velocity_field: Optional[VectorField[FourierField]],
        forward_equation: "NavierStokesVelVortPerturbationInstationary",
        **params: Any,
    ):
        super().__init__(velocity_field, forward_equation, **params)
        self.forward_equation = forward_equation
        self.kappa = self.forward_equation.kappa
        self.accelerate = self.forward_equation.accelerate
        self.n_max = self.forward_equation.n_max

    def update_pressure_gradient(
        self,
        vel_new_field_hat: Optional["jnp_array"] = None,
        dPdx: Optional["float"] = None,
    ) -> "float":
        return 0.0

    def vel_base_fn(self, time_step: int) -> "VectorField[FourierField]":
        return cast(
            "NavierStokesVelVortPerturbationInstationary", self.forward_equation
        ).vel_base_fn(self.number_of_time_steps - time_step)

    def set_linearise(self) -> None:
        try:
            re_ijj_hat = self.get_latest_field("reynolds_stress_ijj_hat").get_data()
        except KeyError:
            re_ijj_hat = VectorField(
                [FourierField.Zeros(self.get_physical_domain()) for _ in range(3)]
            ).get_data()
        self.dPdx = 0.0
        self.source_x_00 = None
        self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_dual_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            self.vel_base_fn(t).get_data(),
            self.get_velocity_u_hat(t),
            re_ijj_hat,
            linearise=False,
            coupling_term=False,
        )

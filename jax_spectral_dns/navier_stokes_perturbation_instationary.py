#!/usr/bin/env python3
from __future__ import annotations

import jax

from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

NoneType = type(None)
# from importlib import reload
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

import jax.numpy as jnp

from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import Field, FourierField, PhysicalField, VectorField
from jax_spectral_dns.navier_stokes_perturbation import (
    update_nonlinear_terms_high_performance_perturbation_skew_symmetric,
)

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        jsd_float,
        jnp_array,
        np_jnp_array,
        Vel_fn_type,
    )


class NavierStokesVelVortPerturbationInstationary(NavierStokesVelVortPerturbation):
    name = "Navier Stokes equation (velocity-vorticity formulation) for perturbations on top of a time-dependent base flow."

    def __init__(self, velocity_field: VectorField[FourierField], **params: Any):

        super().__init__(velocity_field, **params)
        self.kappa = params.get("kappa", 1.0)
        self.accelerate = params.get("accelerate", True)
        self.n_max = params.get("n_max", 100)  # TODO
        print(self.accelerate)

    def vel_base_fn(self, time_step: int) -> "VectorField[FourierField]":
        time_steps = jnp.arange(0, self.end_time, self.get_dt())
        t = time_steps[time_step]
        Re = self.get_Re_tau()

        def g_w(t: jsd_float) -> "jsd_float":
            if self.accelerate:
                return 1.0 - jnp.exp(-self.kappa * t)
            else:
                return jnp.exp(-self.kappa * t)

        def profile_fn(y: "jsd_float") -> "jsd_float":
            d_g_w = jax.grad(g_w)
            out = g_w(t) * y
            out += Re / 6 * d_g_w(t) * (y**3 - y)
            for n in range(1, self.n_max):
                if self.accelerate:
                    c_1_n = (-2 * Re * (-1) ** n / (jnp.pi * n) ** 3) * d_g_w(t) + (
                        2 * jnp.sin(jnp.pi * n) - 2 * jnp.pi * n * jnp.cos(jnp.pi * n)
                    ) / (jnp.pi * n) ** 2
                else:
                    c_1_n = 0.0
                a_n = (jnp.pi * n) ** 2 / Re
                coeff = -1.0 if self.accelerate else 1.0
                int = self.kappa**2 * coeff / (a_n - self.kappa)
                out += (jnp.exp(-self.kappa * t) - jnp.exp(-a_n * t)) * (
                    -(2 * Re * (-1) ** n)
                    / (jnp.pi * n) ** 3
                    * (int + c_1_n)
                    * jnp.sin(n * jnp.pi * y)
                )
            return out

        out_field_x = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: profile_fn(X[1]) + 0.0 * X[2]
        )
        out_field = VectorField(
            [
                out_field_x,
                PhysicalField.Zeros(self.get_physical_domain()),
                PhysicalField.Zeros(self.get_physical_domain()),
            ]
        )
        out_field_hat = out_field.hat()
        out_field_hat.set_name("velocity_base_hat")
        return out_field_hat

    def update_pressure_gradient(
        self,
        vel_new_field_hat: Optional["jnp_array"] = None,
        dPdx: Optional["float"] = None,
    ) -> "float":
        return 0.0

    def set_linearise(self) -> None:
        self.flow_rate = self.get_flow_rate()
        self.dPdx = 0.0
        self.source_x_00 = None
        self.source_z_00 = None

        # self.nonlinear_update_fn = lambda vel, _: update_nonlinear_terms_high_performance_perturbation_rotational(
        self.nonlinear_update_fn = lambda vel, t: update_nonlinear_terms_high_performance_perturbation_skew_symmetric(
            self.get_physical_domain(),
            self.get_domain(),
            vel,
            self.vel_base_fn(t).get_data(),
            linearise=False,
            coupling_term=False,
        )

#!/usr/bin/env python3

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
import jax_cfd.base as cfd
from jax_cfd.base import forcings
from jax_cfd.base import grids
# from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils



TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]


@dataclasses.dataclass
class NavierStokes3D(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.
  """
  viscosity: float
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, self.ky, self.kz = self.grid.rfft_mesh()
    self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2 + self.kz**2)
    # self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    self.linear_term = self.viscosity * self.laplace # WARNING assumes constant viscosity

  def explicit_terms(self, vxhat, vyhat, vzhat):
    vx, vy, vz = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat), jnp.fft.irfftn(vzhat)

    # grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
    # grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
    # grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)
    grad_vxx, grad_vxy, grad_vxz = cfd.finite_differences.central_difference(vx, 0),cfd.finite_differences.central_difference(vx, 1),  cfd.finite_differences.central_difference(vx, 2)
    grad_vyx, grad_vyy, grad_vyz = cfd.finite_differences.central_difference(vy, 0),cfd.finite_differences.central_difference(vy, 1),  cfd.finite_differences.central_difference(vy, 2)
    grad_vzx, grad_vzy, grad_vzz = cfd.finite_differences.central_difference(vz, 0),cfd.finite_differences.central_difference(vz, 1),  cfd.finite_differences.central_difference(vz, 2)
    grad_x = jnp.array([grad_vxx, grad_vyx, grad_vzx])
    grad_y = jnp.array([grad_vxy, grad_vyy, grad_vzy])
    grad_z = jnp.array([grad_vxz, grad_vyz, grad_vzz])

    advection = -(grad_x * vx + grad_y * vy + grad_z * vz)
    advection_hat = jnp.fft.rfftn(advection)

    # if self.smooth is not None:
    #   advection_hat *= self.filter_

    terms = advection_hat

    return terms

  def implicit_terms(self, vxhat, vyhat, vzhat):
    return self.linear_term * jnp.array(vxhat, vyhat, vzhat)

  def implicit_solve(self, vxhat, vyhat, vzhat, time_step):
    return 1 / (1 - time_step * self.linear_term) * jnp.array(vxhat, vyhat, vzhat)

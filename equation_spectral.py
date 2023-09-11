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
    self.laplace = (jnp.pi * 2j)**3 * (self.kx**2 + self.ky**2 + self.kz**2)
    # self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    self.linear_term = self.viscosity * self.laplace

  def explicit_terms(self, vhat):
    vxhat, vyhat, vzhat = vhat
    vx, vy, vz = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat), jnp.fft.irfftn(vzhat)
    print(vxhat.shape)
    print(vx.shape)

    grad_x_hat = 2j * jnp.pi * self.kx * vxhat
    grad_y_hat = 2j * jnp.pi * self.ky * vyhat
    grad_z_hat = 2j * jnp.pi * self.kz * vzhat
    grad_x, grad_y, grad_z = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat), jnp.fft.irfftn(grad_z_hat)

    # print(grad_x.shape)
    # print(vx.shape)
    # print(grad_y.shape)
    # print(vy.shape)
    # print(grad_z.shape)
    # print(vz.shape)
    advection = -(grad_x * vx + grad_y * vy + grad_z * vz)
    # print(advection.shape)
    advection_hat = jnp.fft.rfftn(advection)

    # print(advection_hat.shape)
    # if self.smooth is not None:
    #   advection_hat *= self.filter_

    terms = advection_hat

    return terms

  def implicit_terms(self, vhat):
    vxhat, vyhat, vzhat = vhat
    return self.linear_term * jnp.array(vxhat, vyhat, vzhat)

  def implicit_solve(self, vhat, time_step):
    vxhat, vyhat, vzhat = vhat
    return 1 / (1 - time_step * self.linear_term) * jnp.array(vxhat, vyhat, vzhat)

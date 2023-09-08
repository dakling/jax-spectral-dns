#!/usr/bin/env python3

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

import jax_cfd.base.finite_differences as fd

from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import time_stepping
from jax_cfd.base.equations import stable_time_step, dynamic_time_step, sum_fields, _wrap_term_as_vector
import tree_math

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]


def navier_stokes_pertubation_explicit_terms(
    v_base: GridVariableVector,
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    diffuse: DiffuseFn = diffusion.diffuse,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

  v_base_vec = tree_math._src.vector.Vector(v_base)
  def convect(v):  # pylint: disable=function-redefined
    return tuple(
        advection.advect_van_leer(u, v, dt) for u in v)

  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  def source(v: GridVariableVector) -> GridArrayVector:
    """Returns the source term appearing due to subtraction of mean equation."""
    dim = len(v)
    ret = tuple(grids.GridArray(jnp.sum(jnp.array([fd.central_difference(v_base[i], j).data * v[j].data for j in range(dim)]), axis=0),
                                 offset=v[i].offset,
                                 grid=v[i].grid)
                 for i in range(dim))
    return ret
  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  source_ = _wrap_term_as_vector(source, name='source')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    dv_dt += convection(v_base_vec)
    dv_dt += source_(v)
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs


# TODO(shoyer): rename this to explicit_diffusion_navier_stokes
def semi_implicit_navier_stokes_pertubation(
    v_base: GridVariableVector,
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    time_stepper: Callable = time_stepping.forward_euler,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = navier_stokes_pertubation_explicit_terms(
      v_base=v_base,
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      diffuse=diffuse)

  pressure_projection = jax.named_call(pressure.projection, name='pressure')

  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve)
  )
  step_fn = time_stepper(ode, dt)
  return step_fn

# def implicit_diffusion_navier_stokes(
#     v_base: tuple,
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     diffusion_solve: Callable = diffusion.solve_fast_diag,
#     pressure_solve: Callable = pressure.solve_fast_diag,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""
#   del grid  # unused
#   def convect(v):  # pylint: disable=function-redefined
#     return tuple(
#         advection.advect_van_leer(u, v, dt) for u in v)

#   convect = jax.named_call(convect, name='convection')
#   pressure_projection = jax.named_call(pressure.projection, name='pressure')
#   diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

  # TODO(shoyer): refactor to support optional higher-order time integators
  @jax.named_call
  def navier_stokes_step(v: GridVariableVector) -> GridVariableVector:
    """Computes state at time `t + dt` using first order time integration."""
    convection = convect(v)
    accelerations = [convection]
    dvdt = sum_fields(*accelerations)
    # Update v by taking a time step
    v = tuple(
        grids.GridVariable(u.array + dudt * dt, u.bc)
        for u, dudt in zip(v, dvdt))
    # Pressure projection to incompressible velocity field
    v = pressure_projection(v, pressure_solve)
    # Solve for implicit diffusion
    v = diffusion_solve(v, viscosity, dt)
    return v
  return navier_stokes_step

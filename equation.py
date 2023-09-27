#!/usr/bin/env python3

import functools
from typing import Callable, Optional

from types import NoneType
import jax
import jax.numpy as jnp

class Equation():
  name = "equation"
  def __init__(self, domain, *fields):
    self.domain = domain
    self.fields = {}
    for field in fields:
      f_name = field.name
      self.fields[f_name] = [field]
      self.fields[f_name][0].name = f_name + "_0"
  def get_field(self, name, index=None):
        try:
          if type(index) == NoneType:
            return self.fields[name]
          else:
            return self.fields[name][index]
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

  def get_initial_field(self, name):
      return self.get_field(name, 0)

  def get_latest_field(self, name):
      return self.get_field(name, -1)

  def all_dimensions(self):
      return range(self.domain.number_of_dimensions)

  def all_periodic_dimensions(self):
      return [self.all_dimensions()[d] for d in self.all_dimensions() if self.domain.periodic_directions[d]]

  def all_nonperiodic_dimensions(self):
      return [self.all_dimensions()[d] for d in self.all_dimensions() if not self.domain.periodic_directions[d]]
  def perform_time_step(self):
    raise NotImplementedError()
  def solve(self, dt, number_of_steps):
    raise NotImplementedError()
  def plot(self):
    raise NotImplementedError()

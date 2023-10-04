#!/usr/bin/env python3

import functools
from typing import Callable, Optional
import time

from types import NoneType
import jax
import jax.numpy as jnp


class Equation:
    name = "equation"

    def __init__(self, domain, *fields, **params):
        self.domain = domain
        self.fields = {}
        self.time_step = 0
        self.time = 0.0
        self.dt = 0.0
        self.before_time_step_fn = None
        self.after_time_step_fn = None
        try:
          self.end_time = params["end_time"]
        except KeyError:
          self.end_time = None
        try:
          self.max_iter = params["max_iter"]
        except KeyError:
          self.max_iter = None
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

    def set_field(self, name, index, field):
        try:
            self.fields[name][index] = field
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")


    def append_field(self, name, field):
        try:
            self.fields[name].append(field)
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def all_dimensions(self):
        return range(self.domain.number_of_dimensions)

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.domain.periodic_directions[d]
        ]

    def all_nonperiodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.domain.periodic_directions[d]
        ]

    def done(self):
        iteration_done = False
        time_done = False
        if type(self.max_iter) != NoneType:
          iteration_done = self.time_step > self.max_iter
        if type(self.end_time) != NoneType:
          time_done = self.time > self.end_time
        return iteration_done or time_done

    def perform_time_step(self):
        raise NotImplementedError()

    def before_time_step(self):
      if type(self.before_time_step_fn) != NoneType:
        self.before_time_step_fn(self)

    def after_time_step(self):
      if type(self.after_time_step_fn) != NoneType:
        self.after_time_step_fn(self)

    def solve(self):
      while not self.done():
          i = self.time_step
          print("Time Step " + str(i + 1) + ", time: " + str(self.time) + ", dt: " + str(self.dt))
          # print("Time Step " + str(i + 1))
          start_time = time.time()
          self.before_time_step()
          self.perform_time_step()
          self.after_time_step()
          print("Took " + str(time.time() - start_time) + " seconds")


    def plot(self):
        raise NotImplementedError()

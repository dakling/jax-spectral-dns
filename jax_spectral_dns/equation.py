#!/usr/bin/env python3

from functools import partial
from typing import Callable, Optional
import time

import jax
import jax.numpy as jnp

from pathlib import Path
import os

from jax_spectral_dns.field import Field, FourierField
from jax_spectral_dns.fixed_parameters import FixedParameters


NoneType = type(None)

def print_verb(*str, verbosity_level:int=1, debug:bool=False):
    pref = '  ' * verbosity_level
    if Equation.verbosity_level >= verbosity_level:
        if debug:
            print(pref, end=' ')
            for st in str:
                if type(st) is str:
                    print(st, end='')
                else:
                    jax.debug.callback(lambda x: print(x, end=''), st)
            print()
        else:
            print(pref, *str)


class Equation:
    name = "equation"
    write_intermediate_output = False

    # verbosity_level:
    # 0: no output
    # 1: mostly output from examples.py informing the user about what is being done
    # 2: additional helpful output from the solver
    # 3: even more additional output from the solver that is usually nonessential
    verbosity_level:int = 1

    def __init__(self, domain, *fields, **params):
        try:
            dt = params["dt"]
        except KeyError:
            dt = 1e-2
        self.fixed_parameters = FixedParameters(domain, dt)
        self.fields = {}
        self.time_step = 0
        self.time = 0.0
        self.before_time_step_fn = None
        self.after_time_step_fn = None
        self.post_process_fn = None
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
        # self.initialize()

    @classmethod
    def initialize(cls, cleanup=True):
        Field.initialize(cleanup)

    def get_dt(self):
        return self.fixed_parameters.dt

    def get_domain(self):
        return self.fixed_parameters.domain

    def get_field(self, name, index=None):
        try:
            if type(index) == NoneType:
                return self.fields[name]
            else:
                out = self.fields[name][index]
                if index >= 0:
                    out.set_time_step(index)
                else:
                    out.set_time_step(len(self.fields[name]) + index)
                return out
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

    def append_field(self, name, field, in_place=True):
        try:
            if in_place and len(self.fields[name]) > 0:
                self.fields[name][-1] = field
            else:
                self.fields[name].append(field)
            self.fields[name][-1].name = name + "_" + str(self.time_step)
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def add_field(self, name, field=None):
        assert name not in self.fields, "Field " + name + " already exists!"
        if type(field) == NoneType:
            self.fields[name] = []
        else:
            # self.fields[name] = jnp.array([field]) # TODO avoid the use of lists
            self.fields[name] = [field]
            self.fields[name][0].name = name + "_0"

    def activate_jit(self):
        Field.activate_jit_ = True

    def deactivate_jit(self):
        Field.activate_jit_ = False

    def all_dimensions_jnp(self):
        return jnp.arange(self.get_domain().number_of_dimensions)

    def all_dimensions(self):
        return range(self.get_domain().number_of_dimensions)

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.get_domain().periodic_directions[d]
        ]

    def all_nonperiodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.get_domain().periodic_directions[d]
        ]

    def all_periodic_dimensions_jnp(self):
        return [
            self.all_dimensions_jnp()[d]
            for d in self.all_dimensions_jnp()
            if self.get_domain().periodic_directions[d]
        ]

    def all_nonperiodic_dimensions_jnp(self):
        return [
            self.all_dimensions_jnp()[d]
            for d in self.all_dimensions_jnp()
            if not self.get_domain().periodic_directions[d]
        ]

    def done(self):
        iteration_done = False
        time_done = False
        if type(self.max_iter) != NoneType:
            iteration_done = self.time_step > self.max_iter
        if type(self.end_time) != NoneType:
            time_done = self.time >= self.end_time + self.get_dt()
        return iteration_done or time_done

    def perform_time_step(self, _=None):
        raise NotImplementedError()

    def before_time_step(self):
        if type(self.before_time_step_fn) != NoneType:
            self.before_time_step_fn(self)

    def after_time_step(self):
        if type(self.after_time_step_fn) != NoneType:
            self.after_time_step_fn(self)

    def post_process(self):
        if type(self.post_process_fn) != NoneType:
            raise NotImplementedError()

    def prepare(self):
        pass

    def update_time(self):
        self.time += self.get_dt()
        self.time_step += 1
        # for _, field in self.fields.items():
        #     field[-1].time_step = self.time_step

    def solve_scan(self):
        raise NotImplementedError()

    def solve(self):
        self.prepare()

        if Field.activate_jit_:
            msg = "Solving using jit/scan - this offers high performance but "\
                  "intermediate results won't be available until after the "\
                  "calculation finishes. To disable "\
                  "high-performance mode, use the deactivate_jit()-method of the "\
                  "Equation class."

            print_verb(msg, verbosity_level=2)
            start_time = time.time()
            _, number_of_time_steps = self.solve_scan()
            print_verb(
                "Took "
                + str((time.time() - start_time))
                + " seconds for "
                + str(number_of_time_steps)
                + " time steps, or "
                + str((time.time() - start_time) / number_of_time_steps)
                + " seconds per time step.",
                verbosity_level=1
            )
            self.deactivate_jit()

        else:
            msg = "WARNING: Solving without jit/scan - performance will be "\
                  "significantly lower but intermediate results will be available "\
                  "for printing and plotting. Only recommended for testing. "\
                  "To enable high-performance mode, use the "\
                  "activate_jit()-method of the Equation class."
            print(msg)
            while not self.done():
                i = self.time_step
                print_verb(
                    "Time Step "
                    + str(i + 1)
                    + ", time: "
                    + str(self.time)
                    + ", dt: "
                    + str(self.get_dt())
                )
                start_time = time.time()
                self.before_time_step()
                self.perform_time_step()
                self.update_time()
                self.after_time_step()
                print_verb("Took " + str(time.time() - start_time) + " seconds")

    def plot(self):
        raise NotImplementedError()

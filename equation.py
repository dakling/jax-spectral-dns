#!/usr/bin/env python3

from functools import partial
from typing import Callable, Optional
import time

import jax
import jax.numpy as jnp
import tree_math
from jax_cfd.base.funcutils import trajectory

from pathlib import Path
import os

from importlib import reload
import sys

# try:
#     reload(sys.modules["field"])
# except:
#     print("Unable to load Field")
from field import Field, FourierField


NoneType = type(None)


class Equation:
    name = "equation"
    max_dt = 1e10

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
        # self.initialize()

    @classmethod
    def initialize(cls):
        jax.config.update("jax_enable_x64", True)
        newpaths = [Field.field_dir, Field.plotting_dir]
        for newpath in newpaths:
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        # clean plotting dir
        [f.unlink() for f in Path(Field.plotting_dir).glob("*.pdf") if f.is_file()]
        [f.unlink() for f in Path(Field.plotting_dir).glob("*.png") if f.is_file()]
        [f.unlink() for f in Path(Field.plotting_dir).glob("*.mp4") if f.is_file()]


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

    def supress_plotting(self):
        Field.supress_plotting_ = True

    def enable_plotting(self):
        Field.supress_plotting_ = False

    def all_dimensions_jnp(self):
        return jnp.arange(self.domain.number_of_dimensions)

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

    def all_periodic_dimensions_jnp(self):
        return [
            self.all_dimensions_jnp()[d]
            for d in self.all_dimensions_jnp()
            if self.domain.periodic_directions[d]
        ]

    def all_nonperiodic_dimensions_jnp(self):
        return [
            self.all_dimensions_jnp()[d]
            for d in self.all_dimensions_jnp()
            if not self.domain.periodic_directions[d]
        ]


    def done(self):
        iteration_done = False
        time_done = False
        if type(self.max_iter) != NoneType:
            iteration_done = self.time_step > self.max_iter
        if type(self.end_time) != NoneType:
            time_done = self.time >= self.end_time + self.dt
        return iteration_done or time_done

    def perform_time_step(self, _=None):
        raise NotImplementedError()

    def before_time_step(self):
        if type(self.before_time_step_fn) != NoneType:
            self.before_time_step_fn(self)

    def after_time_step(self):
        if type(self.after_time_step_fn) != NoneType:
            self.after_time_step_fn(self)

    def prepare(self):
        pass

    def update_time(self):
        self.time += self.dt
        self.time_step += 1
        # for _, field in self.fields.items():
        #     field[-1].time_step = self.time_step

    def solve_scan(self):
        raise NotImplementedError()

    def solve(self):

        self.prepare()

        if Field.supress_plotting_:
            start_time = time.time()
            _, number_of_time_steps = self.solve_scan()
            print("Took on average " + str((time.time() - start_time)/number_of_time_steps) + " seconds per time step")

        else:
            while not self.done():
                i = self.time_step
                print(
                    "Time Step "
                    + str(i + 1)
                    + ", time: "
                    + str(self.time)
                    + ", dt: "
                    + str(self.dt)
                )
                # print("Time Step " + str(i + 1))
                start_time = time.time()
                self.before_time_step()
                self.perform_time_step()
                self.update_time()
                self.after_time_step()
                print("Took " + str(time.time() - start_time) + " seconds")

    def plot(self):
        raise NotImplementedError()

#!/usr/bin/env python3

from types import NoneType
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

import numpy as np

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    pass
from domain import Domain

class Field():
    def __init__(self, domain, field, name="field"):
        self.domain = domain
        self.field = field
        self.name = name
        self.field_hat = None

    @classmethod
    def FromFunc(cls, domain, func=None, name="field"):
        if not func:
            func = lambda _: 0.0
        field = jnp.array(list(map(lambda *x: func(x), *domain.mgrid)))
        return cls(domain, field, name)

    def __repr__(self) -> str:
        # fig, ax = plt.subplots(1,1)
        # ax.plot(self.mgrid[0], self.field)
        return str(self.field)

    def __add__(self, other):
        if other.name[0] == "-":
            new_name = self.name + " - " + other.name[1:]
        else:
            new_name = self.name + " + " + other.name
        return Field(self.domain, self.field + other.field, name=new_name)

    def __sub__(self, other):
        return self + other*(-1.0)

    def __mul__(self, number):
        if number >= 0:
            new_name = str(number) + self.name
        elif number == 1:
            new_name = self.name
        elif number == -1:
            new_name = "-" + self.name
        else:
            new_name = "(" + str(number) + ") " + self.name
        return Field(self.domain, self.field * number, name=new_name)

    def __abs__(self):
        return jnp.linalg.norm(self.field)/self.number_of_dofs() # TODO use integration or something more sophisticated

    def number_of_dofs(self):
        return math.prod(self.domain.shape)

    def plot_center(self, dimension, *other_fields):
        # TODO generalize to 3D
        fig, ax = plt.subplots(1,1)
        N_c = len(self.domain.grid[dimension]) // 2
        other_dim = [ i for i in self.all_dimensions() if i != dimension ][0] # TODO only for 2D
        ax.plot(self.domain.grid[dimension], self.field.take(indices=N_c, axis=other_dim), label=self.name)
        for other_field in other_fields:
            ax.plot(self.domain.grid[dimension], other_field.field.take(indices=N_c, axis=other_dim), label=other_field.name)
        fig.legend()
        fig.savefig("plot_cl_" + self.name + "_" + ["x", "y", "z"][dimension] + ".pdf")

    def all_dimensions(self):
        return range(self.domain.number_of_dimensions)

    def all_periodic_dimensions(self):
        return [self.all_dimensions()[d] for d in self.all_dimensions() if self.domain.periodic_directions[d]]

    def all_nonperiodic_dimensions(self):
        return [self.all_dimensions()[d] for d in self.all_dimensions() if not self.domain.periodic_directions[d]]

    def pad_mat_with_zeros(self):
        return jnp.block([[jnp.zeros((1, self.field.shape[1]+2))], [jnp.zeros((self.field.shape[0], 1)), self.field, jnp.zeros((self.field.shape[0], 1))], [jnp.zeros((1, self.field.shape[1]+2))]])

    def update_boundary_conditions(self):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        for dim in self.all_nonperiodic_dimensions():
            self.field = jnp.take(self.field, jnp.array(list(range(len(self.domain.grid[dim]))))[1:-1], axis=dim)
            self.field = jnp.pad(self.field, [(0, 0) if self.domain.periodic_directions[d] else (1,1) for d in self.all_dimensions()], mode="constant", constant_values=0.0)

    # def hat(self):
    #     self.field_hat = FourierField(self)
    #     return self.field_hat

    def diff(self, direction, order=1):
        name_suffix = "".join([["x", "y", "z"][direction] for _ in range(order)])
        return Field(self.domain, self.domain.diff(self.field, direction, order), self.name + "_" + name_suffix)

# class FourierField(Field):
#     def __init__(self, field):
#         self.domain = field.domain # TODO
#         self.name = field.name + "_hat"
#         self.field = field.field
#         for i in range(self.domain.number_of_dimensions):
#             if self.domain.periodic_directions[i]:
#                 self.field = jnp.fft.fft(self.field, axis=i)

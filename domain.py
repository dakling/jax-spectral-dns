#!/usr/bin/env python3

from types import NoneType
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

import numpy as np

class Domain():
    def __init__(self, shape, periodic_directions=None):
        self.number_of_dimensions = len(shape)
        self.grid = []
        self.cheb_mat = None
        self.fourier_mat = None
        self.diff_mats = []
        for dim in range(self.number_of_dimensions):
            if type(periodic_directions) != NoneType and periodic_directions[dim]:
                self.grid.append(self.get_fourier_grid(shape[dim]))
                self.cheb_mat = self.assemble_cheb_diff_mat(shape[dim])
                self.diff_mats[dim] = self.assemble_diff_mat(dim, self.cheb_mat)
            else:
                self.grid.append(self.get_cheb_grid(shape[dim]))
                self.fourier_mat = self.assemble_fourier_diff_mat(shape[dim])
                self.diff_mats[dim] = self.assemble_diff_mat(dim, self.fourier_mat)

    def get_cheb_grid(self, N):
        return jnp.array([jnp.cos(jnp.pi / (N-1) * i) for i in range(N)]) # gauss-lobatto points with endpoints

    def get_fourier_grid(self, N):
        if N % 2 != 0:
            print("Warning: Only even number of points supported for Fourier basis, making the domain larger by one.")
            N += 1
        return jnp.linspace(0.0, 2*jnp.pi, N+1)[:-1]

    def assemble_cheb_diff_mat(self, N, order=1):
        xs = self.get_cheb_grid(N)
        c = jnp.block([2.0, jnp.ones((1, N-2))*1.0, 2.0]) * (-1)**jnp.arange(0,N)
        X = jnp.repeat(xs, N).reshape(N, N)
        dX = X - jnp.transpose(X)
        D_ = jnp.transpose((jnp.transpose(1/c) @ c)) / (dX + jnp.eye(N))
        return D_ - jnp.diag(sum(jnp.transpose(D_)))

    def assemble_fourier_diff_mat(self, N, order=1):
        if N % 2 != 0:
            raise Exception("Fourier discretization points must be even!")
        h = 2*jnp.pi/N;
        column = jnp.block([0, .5*(-1)**jnp.arange(1,N) * 1/jnp.tan(jnp.arange(1,N) * h/2)])
        column2 = jnp.block([column[0], jnp.flip(column[1:])])
        return jnp.linalg.matrix_power(jsc.linalg.toeplitz(column, column2), order)

    def assemble_diff_mat(self, dim, mat):
        def nkron(mat_list):
            out = mat_list[0]
            if len(mat_list > 1):
                for mat in mat_list[1:]:
                    out = jnp.kron(out, mat)
            return out
        return nkron([mat if i == dim else jnp.eye(len(self.grid[i])) for i in range(self.number_of_dimensions)])

    def diff(self, field, direction):
        return self.diff_mats[direction] * field

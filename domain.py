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


class Domain:
    def __init__(self, shape, periodic_directions=None):
        self.number_of_dimensions = len(shape)
        self.periodic_directions = periodic_directions or [
            False for _ in range(self.number_of_dimensions)
        ]
        self.shape = shape
        grid = []
        self.diff_mats = []
        for dim in range(self.number_of_dimensions):
            if type(periodic_directions) != NoneType and self.periodic_directions[dim]:
                grid.append(self.get_fourier_grid(shape[dim]))
                self.diff_mats.append(self.assemble_fourier_diff_mat(shape[dim]))
            else:
                grid.append(self.get_cheb_grid(shape[dim]))
                self.diff_mats.append(self.assemble_cheb_diff_mat(shape[dim]))
        self.grid = jnp.array(grid)
        self.mgrid = jnp.meshgrid(*self.grid, indexing="ij")

    def all_dimensions(self):
        return range(self.number_of_dimensions)

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.periodic_directions[d]
        ]

    def all_nonperiodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.periodic_directions[d]
        ]

    def get_cheb_grid(self, N):
        return jnp.array(
            [jnp.cos(jnp.pi / (N - 1) * i) for i in range(N)]
        )  # gauss-lobatto points with endpoints

    def get_fourier_grid(self, N):
        if N % 2 != 0:
            print(
                "Warning: Only even number of points supported for Fourier basis, making the domain larger by one."
            )
            N += 1
        return jnp.linspace(0.0, 2 * jnp.pi, N + 1)[:-1]

    def hat(self):
        def fftshift(inp):
            N = len(inp)
            return jnp.block([inp[N//2:], inp[:N//2]]) - N//2
        Ns = []
        for i in self.all_periodic_dimensions():
            Ns.append(len(self.grid[i]))
        Ns_arr = jnp.array(Ns)
        fourier_grid = jnp.fft.fftn(self.grid, axes=self.all_periodic_dimensions(), norm="ortho")/(2 * jnp.pi) * Ns_arr
        fourier_grid_shifted = jnp.array(list(map(fftshift, fourier_grid)))
        out = FourierDomain(self.shape, self.periodic_directions)
        out.grid = fourier_grid_shifted
        return out

    def assemble_cheb_diff_mat(self, N, order=1):
        xs = self.get_cheb_grid(N)
        c = jnp.block([2.0, jnp.ones((1, N - 2)) * 1.0, 2.0]) * (-1) ** jnp.arange(0, N)
        X = jnp.repeat(xs, N).reshape(N, N)
        dX = X - jnp.transpose(X)
        D_ = jnp.transpose((jnp.transpose(1 / c) @ c)) / (dX + jnp.eye(N))
        return jnp.linalg.matrix_power(D_ - jnp.diag(sum(jnp.transpose(D_))), order)

    def assemble_fourier_diff_mat(self, N, order=1):
        if N % 2 != 0:
            raise Exception("Fourier discretization points must be even!")
        h = 2 * jnp.pi / N
        column = jnp.block(
            [0, 0.5 * (-1) ** jnp.arange(1, N) * 1 / jnp.tan(jnp.arange(1, N) * h / 2)]
        )
        column2 = jnp.block([column[0], jnp.flip(column[1:])])
        return jnp.linalg.matrix_power(jsc.linalg.toeplitz(column, column2), order)

    def diff(self, field, direction, order=1):
        inds = "ijk"
        diff_mat_ind = "l" + inds[direction]
        other_inds = "".join(
            [
                ind
                for ind in inds[0 : self.number_of_dimensions]
                if ind != inds[direction]
            ]
        )
        target_inds = other_inds[:direction] + "l" + other_inds[direction:]
        field_ind = inds[0 : self.number_of_dimensions]
        ind = field_ind + "," + diff_mat_ind + "->" + target_inds
        f_diff = jnp.einsum(
            ind, field, jnp.linalg.matrix_power(self.diff_mats[direction], order)
        )
        return f_diff


class FourierDomain(Domain):
    pass

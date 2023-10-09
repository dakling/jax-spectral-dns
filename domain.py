#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

import numpy as np

NoneType = type(None)


class Domain:
    def __init__(self, shape, periodic_directions=None, scale_factors=None):
        self.number_of_dimensions = len(shape)
        self.periodic_directions = periodic_directions or [
            False for _ in jnp.arange(self.number_of_dimensions)
        ]
        if type(scale_factors) == NoneType:
            self.scale_factors = []
        else:
            self.scale_factors = scale_factors
        self.shape = shape
        grid = []
        self.diff_mats = []
        for dim in jnp.arange(self.number_of_dimensions):
            if type(periodic_directions) != NoneType and self.periodic_directions[dim]:
                if type(scale_factors) == NoneType:
                    self.scale_factors.append(2.0 * jnp.pi)
                grid.append(self.get_fourier_grid(shape[dim], self.scale_factors[dim]))
                self.diff_mats.append(
                    self.assemble_fourier_diff_mat(shape[dim])
                    * (2 * jnp.pi)
                    / self.scale_factors[dim]
                )
            else:
                if type(scale_factors) == NoneType:
                    self.scale_factors.append(1.0)
                grid.append(self.get_cheb_grid(shape[dim], self.scale_factors[dim]))
                self.diff_mats.append(self.assemble_cheb_diff_mat(shape[dim]))
        # self.grid = jnp.array(grid)
        self.grid = grid
        self.mgrid = jnp.meshgrid(*self.grid, indexing="ij")

    def all_dimensions(self):
        return range(self.number_of_dimensions)
        # return jnp.arange(self.number_of_dimensions)

    def is_periodic(self, direction):
        return self.periodic_directions[direction]

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.is_periodic(d)
        ]

    def all_nonperiodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.is_periodic(d)
        ]

    def get_cheb_grid(self, N, scale_factor=1.0):
        assert (
            scale_factor == 1.0
        ), "different scaling of Chebyshev direction not implemented yet."
        return jnp.array(
            [jnp.cos(jnp.pi / (N - 1) * i) for i in jnp.arange(N)]
        )  # gauss-lobatto points with endpoints

    def get_fourier_grid(self, N, scale_factor=2 * jnp.pi):
        if N % 2 != 0:
            print(
                "Warning: Only even number of points supported for Fourier basis, making the domain larger by one."
            )
            N += 1
        return jnp.linspace(0.0, scale_factor, N + 1)[:-1]

    def hat(self):
        def fftshift(inp, i):
            if self.periodic_directions[i]:
                N = len(inp)
                return (
                    (jnp.block([inp[N // 2 :], inp[: N // 2]]) - N // 2)
                    * (2 * jnp.pi)
                    / self.scale_factors[i]
                )
            else:
                return inp

        Ns = []
        for i in self.all_dimensions():
            Ns.append(len(self.grid[i]))
        fourier_grid = []
        for i in self.all_dimensions():
            if self.periodic_directions[i]:
                fourier_grid.append(jnp.linspace(0, Ns[i] - 1, Ns[i]))
            else:
                fourier_grid.append(self.grid[i])
        fourier_grid_shifted = (
            list(map(fftshift, fourier_grid, self.all_dimensions()))
        )
        out = FourierDomain(
            self.shape, self.periodic_directions, scale_factors=self.scale_factors
        )
        out.grid = fourier_grid_shifted
        out.mgrid = jnp.meshgrid(*fourier_grid_shifted, indexing="ij")
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

    def diff_fourier_field_slice(self, field, direction, order=1):
        # inds = "ijk"
        # diff_mat_ind = "l" + inds[0]
        # other_inds = "".join(
        #     [
        #         ind
        #         for ind in inds[0 : 1]
        #         if ind != inds[0]
        #     ]
        # )
        # target_inds = other_inds[:0] + "l" + other_inds[0:]
        # field_ind = inds[0 : 1]
        # ind = field_ind + "," + diff_mat_ind + "->" + target_inds
        # f_diff = jnp.einsum(
        #     ind, field, jnp.linalg.matrix_power(self.diff_mats[direction], order)
        # )
        return jnp.linalg.matrix_power(self.diff_mats[direction], order) @ field
        # return f_diff

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction):
        def set_first_mat_row_and_col_to_unit(matr):
            N = matr.shape[0]
            return jnp.block(
                [[1, jnp.zeros((1, N - 1))], [jnp.zeros((N - 1, 1)), matr[1:, 1:]]]
            )

        def set_last_mat_row_and_col_to_unit(matr):
            N = matr.shape[0]
            return jnp.block(
                [[matr[:-1, :-1], jnp.zeros((N - 1, 1))], [jnp.zeros((1, N - 1)), 1]]
            )

        mat = set_last_mat_row_and_col_to_unit(
            set_first_mat_row_and_col_to_unit(
                jnp.linalg.matrix_power(self.diff_mats[direction], 2)
            )
        )
        return mat

    def integrate(self, field, direction, order=1, bc_left=None, bc_right=None):
        if (type(bc_left) != NoneType and abs(bc_left) > 1e-20) or (
            type(bc_right) != NoneType and abs(bc_right) > 1e-20
        ):
            raise Exception("Only homogeneous dirichlet conditions currently supported")
        assert order <= 2, "Integration only supported up to second order"

        def set_first_mat_row_and_col_to_unit(matr):
            if bc_right == None:
                return matr
            N = matr.shape[0]
            return jnp.block(
                [[1, jnp.zeros((1, N - 1))], [jnp.zeros((N - 1, 1)), matr[1:, 1:]]]
            )

        def set_last_mat_row_and_col_to_unit(matr):
            if bc_left == None:
                return matr
            N = matr.shape[0]
            return jnp.block(
                [[matr[:-1, :-1], jnp.zeros((N - 1, 1))], [jnp.zeros((1, N - 1)), 1]]
            )

        def set_first_and_last_of_field(field, first, last):
            N = field.shape[direction]
            inds = jnp.arange(1, N - 1)
            inner = field.take(indices=inds, axis=direction)
            out = jnp.pad(
                inner,
                [(0, 0) if d != direction else (1, 1) for d in self.all_dimensions()],
                mode="constant",
                constant_values=(first, last),
            )
            return out

        mat = set_last_mat_row_and_col_to_unit(
            set_first_mat_row_and_col_to_unit(
                jnp.linalg.matrix_power(self.diff_mats[direction], order)
            )
        )
        inv_mat = jnp.linalg.inv(mat)
        # b_right = 0.0 if type(bc_right) != NoneType else b_right_fallback
        # b_left = 0.0 if type(bc_left) != NoneType else b_left_fallback
        b_right = 0.0
        b_left = 0.0
        b = set_first_and_last_of_field(field, b_right, b_left)

        inds = "ijk"
        int_mat_ind = "l" + inds[direction]
        other_inds = "".join(
            [
                ind
                for ind in inds[0 : self.number_of_dimensions]
                if ind != inds[direction]
            ]
        )
        target_inds = other_inds[:direction] + "l" + other_inds[direction:]
        field_ind = inds[0 : self.number_of_dimensions]
        ind = field_ind + "," + int_mat_ind + "->" + target_inds
        out = jnp.einsum(ind, b, inv_mat)

        # out_right = bc_right if type(bc_right) != NoneType else out[0]
        # out_left = bc_left if type(bc_left) != NoneType else out[-1]
        # out_right = 0.0
        # out_left = 0.0
        # # out_bc = set_first_and_last_of_field(out, out_right, out_left)
        out_bc = out
        return out_bc

    def solve_poisson_fourier_field_slice(self, field, mat, k1, k2):
        mat_inv = mat[k1-1, k2-1, :, :]
        rhs_hat = field
        out_field = mat_inv @ rhs_hat
        return out_field

    def update_boundary_conditions_fourier_field_slice(self, field, non_periodic_direction):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        out_field = jnp.take(
            field,
            jnp.arange(len(self.grid[non_periodic_direction]))[1:-1],
            axis=0,
        )
        out_field = jnp.pad(
            out_field,
            [
                (1,1)
            ],
            mode="constant",
            constant_values=0.0,
        )
        return out_field

    def no_hat(self, field):
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        return jnp.fft.ifftn(
            field, axes=self.all_periodic_dimensions(), norm="ortho"
        ).real / (1 / scaling_factor)

    def field_hat(self, field):
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        return (jnp.fft.fftn(field, axes=list(self.all_periodic_dimensions()), norm="ortho") / scaling_factor)

    def curl(self, field):
        assert len(field) == 3, "rotation only defined in 3 dimensions"
        u_y = self.diff(field[0], (1))
        u_z = self.diff(field[0], (2))
        v_x = self.diff(field[1], (0))
        v_z = self.diff(field[1], (2))
        w_x = self.diff(field[2], (0))
        w_y = self.diff(field[2], (1))

        curl_0 = w_y - v_z
        curl_1 = u_z - w_x
        curl_2 = v_x - u_y

        return [curl_0, curl_1, curl_2]

    def cross_product(self, field_1, field_2):
        out_0 = field_1[2] * field_2[1] - field_1[1] * field_2[2]
        out_1 = field_1[0] * field_2[2] - field_1[2] * field_2[0]
        out_2 = field_1[1] * field_2[0] - field_1[0] * field_2[1]
        return [out_0, out_1, out_2]


class FourierDomain(Domain):
    pass

#!/usr/bin/env python3

from abc import ABC
import jax
import jax.numpy as jnp
import numpy as np
# import jax.scipy as jsc
import scipy as sc
from jax.tree_util import register_pytree_node_class
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc
from functools import partial
import dataclasses
from typing import Tuple, Union, Sequence, List



NoneType = type(None)


def get_cheb_grid(N, scale_factor=1.0):
    """Assemble a Chebyshev grid with N points on the interval [-1, 1],
    unless scaled to a different interval using scale_factor (currently not
    implemented)."""
    assert (
        scale_factor == 1.0
    ), "different scaling of Chebyshev direction not implemented yet."
    # n = int(N * self.aliasing)
    n = N
    return np.array(
        [np.cos(np.pi / (n - 1) * i) for i in np.arange(n)]
    )  # gauss-lobatto points with endpoints


def get_fourier_grid(N, scale_factor=2 * np.pi, aliasing=1.0):
    """Assemble a Fourier grid (equidistant) with N points on the interval [0, 2pi],
    unless scaled to a different interval using scale_factor."""
    if N % 2 != 0:
        print(
            "Warning: Only even number of points supported for Fourier basis, making the domain larger by one."
        )
        N += 1
    return np.linspace(start=0.0, stop=scale_factor, num=int(N * aliasing + 1))[:-1]


def assemble_cheb_diff_mat(xs, order=1):
    """Assemble a 1D Chebyshev differentiation matrix in direction i with
    differentiation order order."""
    N = len(xs)
    c = np.block([2.0, np.ones((1, N - 2)) * 1.0, 2.0]) * (-1) ** np.arange(0, N)
    X = np.repeat(xs, N).reshape(N, N)
    dX = X - np.transpose(X)
    D_ = np.transpose((np.transpose(1 / c) @ c)) / (dX + np.eye(N))
    return np.linalg.matrix_power(D_ - np.diag(sum(np.transpose(D_))), order)


def assemble_fourier_diff_mat(n, order=1):
    """Assemble a 1D Fourier differentiation matrix in direction i with
    differentiation order order."""
    if n % 2 != 0:
        raise Exception("Fourier discretization points must be even!")
    h = 2 * np.pi / n
    column = np.block(
        [0, 0.5 * (-1) ** np.arange(1, n) * 1 / np.tan(np.arange(1, n) * h / 2)]
    )
    column2 = np.block([column[0], np.flip(column[1:])])
    return np.linalg.matrix_power(sc.linalg.toeplitz(column, column2), order)


# @register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Domain(ABC):
    """Class that mainly contains information on the independent variables of
    the problem (i.e. the basis) and implements some operations that can be
    performed on it."""

    number_of_dimensions: int
    periodic_directions: Tuple[bool]
    scale_factors: Union[List[jnp.float64], Tuple[jnp.float64]]
    shape: Sequence[int]
    grid: List[jnp.ndarray]
    diff_mats: List[jnp.ndarray]
    mgrid: List[jnp.ndarray]
    aliasing: jnp.float64 = 1  # no antialiasing (requires finer resolution)
    # aliasing = 3 / 2  # prevent aliasing using the 3/2-rule

    # def tree_flatten(self):
    #     children = (self.grid, self.diff_mats, self.mgrid)
    #     aux_data = (self.number_of_dimensions, self.periodic_directions, self.scale_factors, self.shape, self.aliasing)
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*aux_data[:4], *children, aux_data[5])

    @classmethod
    def create(
        cls,
        shape: Sequence[int],
        periodic_directions: Sequence[bool],
        scale_factors: Union[List[np.float64], Tuple[np.float64], NoneType] = None,
        aliasing=1,
        _grid: Union[NoneType, List[np.ndarray]] = None,
        _mgrid: Union[NoneType, List[np.ndarray]] = None,
    ):
        number_of_dimensions = len(shape)
        if type(scale_factors) == NoneType:
            scale_factors_: Union[List[np.float64]] = []
        else:
            assert isinstance(scale_factors, List) or isinstance(scale_factors, Tuple)
            scale_factors_ = scale_factors
        if type(_grid) == NoneType:
            grid = []
        else:
            grid = _grid
        diff_mats = []
        for dim in range(number_of_dimensions):
            if type(periodic_directions) != NoneType and periodic_directions[dim]:
                if type(scale_factors) == NoneType:
                    scale_factors_.append(2.0 * np.pi)
                if type(_grid) == NoneType:
                    grid.append(
                        get_fourier_grid(shape[dim], scale_factors_[dim], aliasing)
                    )
                diff_mats.append(
                    assemble_fourier_diff_mat(n=shape[dim], order=1)
                    * (2 * np.pi)
                    / scale_factors_[dim]
                )
            else:
                if type(scale_factors) == NoneType:
                    scale_factors_.append(1.0)
                if type(_grid) == NoneType:
                    grid.append(get_cheb_grid(shape[dim], scale_factors_[dim]))
                diff_mats.append(assemble_cheb_diff_mat(grid[dim]))
        if type(_mgrid) == NoneType:
            mgrid = np.meshgrid(*grid, indexing="ij")
        else:
            mgrid = _mgrid
        return cls(
            number_of_dimensions,
            periodic_directions,
            scale_factors_,
            shape,
            grid,
            diff_mats,
            mgrid,
            aliasing,
        )

    def number_of_cells(self, direction):
        return len(self.grid[direction])

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

    def hat(self):
        """Create a Fourier transform of the present domain in all periodic
        directions and return the resulting domain."""

        def fftshift(inp, i):
            if self.periodic_directions[i]:
                N = len(inp)
                return (
                    (np.block([inp[N // 2 :], inp[: N // 2]]) - N // 2)
                    * (2 * np.pi)
                    / self.scale_factors[i]
                )
            else:
                return inp

        Ns = []
        for i in self.all_dimensions():
            Ns.append(self.number_of_cells(i) / self.aliasing)
        fourier_grid = []
        for i in self.all_dimensions():
            if self.periodic_directions[i]:
                fourier_grid.append(np.linspace(0, Ns[i] - 1, int(Ns[i])))
            else:
                fourier_grid.append(self.grid[i])
        fourier_grid_shifted = list(map(fftshift, fourier_grid, self.all_dimensions()))
        grid = fourier_grid_shifted
        mgrid = np.meshgrid(*fourier_grid_shifted, indexing="ij")
        out = FourierDomain.create(
            self.shape,
            self.periodic_directions,
            scale_factors=self.scale_factors,
            _grid=grid,
            _mgrid=mgrid,
        )
        return out

    # @partial(jax.jit, static_argnums=(0,2,3))
    def diff(self, field, direction, order=1):
        """Calculate and return the derivative of given order for field in
        direction."""
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
            ind, field, np.linalg.matrix_power(self.diff_mats[direction], order)
        )
        return f_diff

    def diff_fourier_field_slice(self, field, direction, order=1):
        """Calculate and return the derivative of given order for a Fourier
        field slice in direction."""
        return np.linalg.matrix_power(self.diff_mats[direction], order) @ field

    def enforce_homogeneous_dirichlet(self, mat):
        """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        homogeneous dirichlet boundary conditions at both ends by setting the
        off-diagonal elements of its first and last rows and columns to zero and
        the diagonal elements to unity."""

        def set_first_mat_row_and_col_to_unit(matr):
            N = matr.shape[0]
            return np.block(
                [[1, np.zeros((1, N - 1))], [np.zeros((N - 1, 1)), matr[1:, 1:]]]
            )

        def set_last_mat_row_and_col_to_unit(matr):
            N = matr.shape[0]
            return np.block(
                [[matr[:-1, :-1], np.zeros((N - 1, 1))], [np.zeros((1, N - 1)), 1]]
            )

        return set_last_mat_row_and_col_to_unit(set_first_mat_row_and_col_to_unit(mat))

    def enforce_inhomogeneous_dirichlet(self, mat, rhs, bc_left, bc_right):
        # """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        # inhomogeneous dirichlet boundary conditions at both ends by setting the
        # off-diagonal elements of its first and last rows to zero and
        # the diagonal elements to unity, and the first and last element of the
        # rhs to the desired values bc_left and bc_right."""
        def set_first_mat_row_to_unit(matr):
            N = matr.shape[0]
            return np.block([[1, np.zeros((1, N - 1))], [matr[1:, :]]])

        def set_last_mat_row_to_unit(matr):
            N = matr.shape[0]
            return np.block([[matr[:-1, :]], [np.zeros((1, N - 1)), 1]])

        out_mat = set_last_mat_row_to_unit(set_first_mat_row_to_unit(mat))
        out_rhs = np.block([bc_right, rhs[1:-1], bc_left])

        return (out_mat, out_rhs)

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction):
        """Assemble the Chebyshev differentiation matrix of second order with
        homogeneous Dirichlet boundary conditions enforced by setting the first
        and last rows and columns to one (diagonal elements) and zero
        (off-diagonal elements)"""

        return self.enforce_homogeneous_dirichlet(
            np.linalg.matrix_power(self.diff_mats[direction], 2)
        )

    def integrate(self, field, direction, order=1, bc_left=None, bc_right=None):
        """Calculate the integral or order for field in direction subject to the
        boundary conditions bc_left and/or bc_right. Since this is difficult to
        generalize, only cases that are needed are implemented."""
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

        def set_first_of_field(field, first):
            N = field.shape[direction]
            inds = jnp.arange(1, N)
            inner = field.take(indices=inds, axis=direction)
            out = jnp.pad(
                inner,
                [(0, 0) if d != direction else (1, 0) for d in self.all_dimensions()],
                mode="constant",
                constant_values=first,
            )
            return out

        def set_last_of_field(field, first):
            N = field.shape[direction]
            inds = jnp.arange(0, N - 1)
            inner = field.take(indices=inds, axis=direction)
            out = jnp.pad(
                inner,
                [(0, 0) if d != direction else (0, 1) for d in self.all_dimensions()],
                mode="constant",
                constant_values=first,
            )
            return out

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

        if not self.is_periodic(direction):
            if order == 1:
                if type(bc_right) != NoneType and type(bc_left) == NoneType:
                    mat = set_first_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                    b = set_first_of_field(field, bc_right)
                elif type(bc_left) != NoneType and type(bc_right) == NoneType:
                    mat = set_last_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                    b = set_last_of_field(field, bc_left)

            elif order == 2:
                mat = set_last_mat_row_and_col_to_unit(
                    set_first_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                )
                b_right = 0.0
                b_left = 0.0
                b = set_first_and_last_of_field(field, b_right, b_left)
        else:
            raise Exception(
                "Integration not implemented in periodic directions, use Fourier integration instead."
            )
            mat = set_first_mat_row_and_col_to_unit(
                jnp.linalg.matrix_power(self.diff_mats[direction], order)
            )

            b_right = 0.0
            b_left = 0.0
            b = set_first_of_field(field, b_right)
        inv_mat = jnp.linalg.inv(mat)

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

        out_bc = out
        return out_bc

    def update_boundary_conditions(self, field):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        for dim in self.all_nonperiodic_dimensions():
            field = jnp.take(
                field,
                jnp.arange(self.number_of_cells(dim))[1:-1],
                axis=dim,
            )
            field = jnp.pad(
                field,
                [
                    (0, 0) if self.periodic_directions[d] else (1, 1)
                    for d in self.all_dimensions()
                ],
                mode="constant",
                constant_values=0.0,
            )
        return field

    def solve_poisson_fourier_field_slice(self, field, mat, k1, k2):
        """Solve the poisson equation with field as the right-hand side for a
        one-dimensional slice at the wavenumbers k1 and k2. Use the provided
        differentiation matrix mat."""
        mat_inv = mat[k1, k2, :, :]
        rhs_hat = field
        out_field = mat_inv @ rhs_hat
        return out_field

    def update_boundary_conditions_fourier_field_slice(
        self, field, non_periodic_direction
    ):
        """Set the boundary conditions for a one-dimensional slice of a field
        along the non_periodic_direction. This assumes homogeneous dirichlet
        conditions in all non-periodic directions"""
        out_field = jnp.take(
            field,
            jnp.arange(self.number_of_cells(non_periodic_direction))[1:-1],
            axis=0,
        )
        out_field = jnp.pad(
            out_field,
            [(1, 1)],
            mode="constant",
            constant_values=0.0,
        )
        return out_field

    def no_hat(self, field):
        """Compute the inverse Fourier transform of field."""
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        Ns = [
            int(self.number_of_cells(i) * 1 / self.aliasing)
            for i in self.all_dimensions()
        ]
        ks = [int((Ns[i]) / 2) for i in self.all_dimensions()]
        for i in self.all_periodic_dimensions():
            field_1 = field.take(indices=jnp.arange(0, ks[i]), axis=i)
            field_2 = field.take(indices=jnp.arange(Ns[i] - ks[i], Ns[i]), axis=i)
            zeros_shape = [
                field_1.shape[dim] if dim != i else int(Ns[i] * (self.aliasing - 1))
                for dim in self.all_dimensions()
            ]
            extra_zeros = jnp.zeros(zeros_shape)
            field = jnp.concatenate([field_1, extra_zeros, field_2], axis=i)

        out = jnp.fft.ifftn(
            field,
            axes=self.all_periodic_dimensions(),
            norm="ortho",
        ).real / (1 / scaling_factor)
        return out

    def field_hat(self, field):
        """Compute the Fourier transform of field."""
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        Ns = [self.number_of_cells(i) for i in self.all_dimensions()]
        ks = [
            int((Ns[i] - Ns[i] * (1 - 1 / self.aliasing)) / 2)
            for i in self.all_dimensions()
        ]

        out = (
            jnp.fft.fftn(field, axes=list(self.all_periodic_dimensions()), norm="ortho")
            / scaling_factor
        )

        for i in self.all_periodic_dimensions():
            out_1 = out.take(indices=jnp.arange(0, ks[i]), axis=i)
            out_2 = out.take(indices=jnp.arange(Ns[i] - ks[i], Ns[i]), axis=i)
            out = jnp.concatenate([out_1, out_2], axis=i)

        return out

    # @partial(jax.jit, static_argnums=(0))
    def curl(self, field):
        """Compute the curl of field."""
        # assert len(field) == 3, "rotation only defined in 3 dimensions"
        u_y = self.diff(field[0,...], 1)
        u_z = self.diff(field[0,...], 2)
        v_x = self.diff(field[1,...], 0)
        v_z = self.diff(field[1,...], 2)
        w_x = self.diff(field[2,...], 0)
        w_y = self.diff(field[2,...], 1)

        curl_0 = w_y - v_z
        curl_1 = u_z - w_x
        curl_2 = v_x - u_y

        return jnp.array([curl_0, curl_1, curl_2])

    def cross_product(self, field_1, field_2):
        """Compute the cross (or vector) product of field_1 and field_2."""
        out_0 = field_1[1,...] * field_2[2,...] - field_1[2,...] * field_2[1,...]
        out_1 = field_1[2,...] * field_2[0,...] - field_1[0,...] * field_2[2,...]
        out_2 = field_1[0,...] * field_2[1,...] - field_1[1,...] * field_2[0,...]
        return jnp.array([out_0, out_1, out_2])


@dataclasses.dataclass(frozen=True)
class PhysicalDomain(Domain):
    """Domain that lives in physical space (as opposed to Fourier space)."""

    def __hash__(self):
        return hash(
            (
                self.number_of_dimensions,
                self.periodic_directions,
                self.scale_factors,
                self.shape,
                self.aliasing,
            )
        )


@dataclasses.dataclass(frozen=True, init=False)
class FourierDomain(Domain):
    """Same as Domain but lives in Fourier space."""

    def __hash__(self):
        return hash(
            (
                self.number_of_dimensions,
                self.periodic_directions,
                self.scale_factors,
                self.shape,
                self.aliasing,
            )
        )

    def assemble_poisson_matrix(self):
        assert len(self.all_dimensions()) == 3, "Only 3d implemented currently."
        assert (
            len(self.all_nonperiodic_dimensions()) <= 1
        ), "Poisson solution not implemented for the general case."
        # y_mat = self.get_cheb_mat_2_homogeneous_dirichlet_only_rows(
        y_mat = self.get_cheb_mat_2_homogeneous_dirichlet(
            self.all_nonperiodic_dimensions()[0]
        )
        n = y_mat.shape[0]
        bc_padding = 1
        eye_bc = jnp.block(
            [
                [jnp.zeros((bc_padding, n))],
                [
                    jnp.zeros((n - 2 * bc_padding, bc_padding)),
                    jnp.eye(n - 2 * bc_padding),
                    jnp.zeros((n - 2 * bc_padding, bc_padding)),
                ],
                [jnp.zeros((bc_padding, n))],
            ]
        )
        k1 = self.grid[self.all_periodic_dimensions()[0]]
        k2 = self.grid[self.all_periodic_dimensions()[1]]
        k1sq = k1**2
        k2sq = k2**2
        mat = jnp.array(
            [
                [jnp.linalg.inv((-(k1sq_ + k2sq_)) * eye_bc + y_mat) for k2sq_ in k2sq]
                for k1sq_ in k1sq
            ]
        )
        return mat

    # @partial(jax.jit, static_argnums=(0,2,3))
    def diff(self, field, direction, order=1, physical_domain=None):
        """Calculate and return the derivative of given order for field in
        direction."""
        if direction in self.all_periodic_dimensions():
            f_diff = (1j * self.mgrid[direction]) ** order * field
        else:
            f_diff = physical_domain.diff(field, direction, order)
        return f_diff

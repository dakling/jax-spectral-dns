#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
import math
import jax
import jax.numpy as jnp
from orthax import chebyshev  # type: ignore
import numpy as np
import scipy as sc  # type: ignore
import dataclasses
from typing import (
    TYPE_CHECKING,
    Iterable,
    Optional,
    Tuple,
    Union,
    Sequence,
    List,
    Any,
    cast,
)
from typing_extensions import Self

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        np_float_array,
        np_complex_array,
        jnp_array,
        np_jnp_array,
        jsd_float,
    )

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

NoneType = type(None)

# use_rfftn = jax.default_backend() == "cpu"
# use_rfftn = True
use_rfftn = False
# jit_rfftn = True
jit_rfftn = False
custom_irfftn = True
print("using rfftn?", use_rfftn)
print("jitting rfftn?", jit_rfftn)
print("custom irfftn?", custom_irfftn)


def get_irfftn_data_custom(data: "jnp_array", axes: List[int]) -> "jnp_array":
    rfftn_axis = axes[-1]
    N = data.shape[rfftn_axis]
    inds = jnp.arange(1, N)
    added_data = jnp.flip(
        jnp.conjugate(data.take(indices=inds, axis=rfftn_axis)), axis=rfftn_axis
    )
    first_data = data.take(indices=jnp.arange(0, N - 1), axis=rfftn_axis)
    full_data = jnp.concatenate([first_data, added_data], axis=rfftn_axis)
    out = jnp.fft.ifftn(full_data, axes=axes, norm="ortho")
    return out


if use_rfftn:
    if jit_rfftn:
        rfftn_jit = jax.jit(
            lambda f, dims: jnp.fft.rfftn(f, axes=list(dims), norm="ortho"),
            static_argnums=1,
        )
        if custom_irfftn:
            irfftn_jit = jax.jit(
                lambda f, dims: get_irfftn_data_custom(f, axes=list(dims)),
                static_argnums=1,
            )
        else:
            irfftn_jit = jax.jit(
                lambda f, dims: jnp.fft.irfftn(f, axes=list(dims), norm="ortho"),
                static_argnums=1,
            )
    else:
        rfftn_jit = lambda f, dims: jnp.fft.rfftn(f, axes=list(dims), norm="ortho")

        if custom_irfftn:
            irfftn_jit = lambda f, dims: get_irfftn_data_custom(f, axes=list(dims))
        else:
            irfftn_jit = lambda f, dims: jnp.fft.irfftn(
                f, axes=list(dims), norm="ortho"
            )

else:
    if jit_rfftn:
        rfftn_jit = jax.jit(
            lambda f, dims: jnp.fft.fftn(f, axes=list(dims), norm="ortho"),
            static_argnums=1,
        )

        irfftn_jit = jax.jit(
            lambda f, dims: jnp.fft.ifftn(f, axes=list(dims), norm="ortho"),
            static_argnums=1,
        )
    else:
        rfftn_jit = lambda f, dims: jnp.fft.fftn(f, axes=list(dims), norm="ortho")

        irfftn_jit = lambda f, dims: jnp.fft.ifftn(f, axes=list(dims), norm="ortho")


def get_cheb_grid(N: int, scale_factor: float = 1.0) -> "np_float_array":
    """Assemble a Chebyshev grid with N points on the interval [-1, 1],
    unless scaled to a different interval using scale_factor (currently not
    implemented)."""
    assert (
        scale_factor == 1.0
    ), "different scaling of Chebyshev direction not implemented yet."
    n = int(N)
    return np.array(
        [np.cos(np.pi / (n - 1) * i) for i in np.arange(n)]
    )  # gauss-lobatto points with endpoints


def get_fourier_grid(N: int, scale_factor: float = 2.0 * np.pi) -> "np_float_array":
    """Assemble a Fourier grid (equidistant) with N points on the interval [0, 2pi],
    unless scaled to a different interval using scale_factor."""
    if N % 2 != 0:
        n = int((N - 1))
    else:
        n = int(N)
    if n % 2 != 0:
        raise Exception("Fourier discretization points must be even!")
    return np.linspace(start=0.0, stop=scale_factor, num=int(n + 1))[:-1]


def assemble_cheb_diff_mat(xs: "np_float_array", order: int = 1) -> "np_float_array":
    """Assemble a 1D Chebyshev differentiation matrix in direction i with
    differentiation order order."""
    N = len(xs)
    c = np.block([2.0, np.ones((1, N - 2)) * 1.0, 2.0]) * (-1) ** np.arange(0, N)
    X = np.repeat(xs, N).reshape(N, N)
    dX = X - np.transpose(X)
    D_ = np.transpose((np.transpose(1 / c) @ c)) / (dX + np.eye(N))
    return np.linalg.matrix_power(D_ - np.diag(sum(np.transpose(D_))), order)


def assemble_fourier_diff_mat(N: int, order: int = 1) -> "np_float_array":
    """Assemble a 1D Fourier differentiation matrix in direction i with
    differentiation order order."""
    if N % 2 != 0:
        n = int((N - 1))
    else:
        n = int(N)
    if n % 2 != 0:
        raise Exception("Fourier discretization points must be even!")
    h = 2 * np.pi / n
    column = np.block(
        [0, 0.5 * (-1) ** np.arange(1, n) * 1 / np.tan(np.arange(1, n) * h / 2)]
    )
    column2 = np.block([column[0], np.flip(column[1:])])
    return np.linalg.matrix_power(sc.linalg.toeplitz(column, column2), order)


# @register_pytree_node_class
# @dataclasses.dataclass(frozen=True, kw_only=True)
@dataclasses.dataclass(frozen=True)
class Domain(ABC):
    """Class that mainly contains information on the independent variables of
    the problem (i.e. the basis) and implements some operations that can be
    performed on it."""

    number_of_dimensions: int
    periodic_directions: Tuple[bool, ...]
    scale_factors: Tuple[float, ...]
    shape: Tuple[int, ...]
    grid: Tuple["np_float_array", ...]
    diff_mats: Tuple["np_float_array", ...]
    mgrid: Tuple["np_float_array", ...]
    aliasing: float = 3 / 2  # prevent aliasing using the 3/2-rule
    dealias_nonperiodic: bool = False

    @classmethod
    def create(
        cls,
        shape: Tuple[int, ...],
        periodic_directions: Tuple[bool, ...],
        scale_factors: Optional[Tuple[float, ...]] = None,
        aliasing: float = 3 / 2,
        dealias_nonperiodic: bool = False,
        physical_shape_passed: bool = False,
    ) -> Self:
        number_of_dimensions = len(shape)
        if type(scale_factors) == NoneType:
            scale_factors_: List[float] = []
        else:
            assert isinstance(scale_factors, list) or isinstance(scale_factors, tuple)
            scale_factors_ = list(scale_factors)
        if not physical_shape_passed:
            shape = tuple(
                (
                    int(shape[i])
                    if not periodic_directions[i] or int(shape[i]) % 2 != 0
                    else int(shape[i]) + 1
                )
                for i in range(len(shape))
            )
            physical_shape = tuple(
                (
                    math.ceil(
                        shape[i]
                        * (
                            aliasing
                            if periodic_directions[i] or dealias_nonperiodic
                            else 1
                        )
                    )
                    if not periodic_directions[i]
                    or math.ceil(
                        shape[i]
                        * (
                            aliasing
                            if periodic_directions[i] or dealias_nonperiodic
                            else 1
                        )
                    )
                    % 2
                    != 0
                    else math.ceil(
                        shape[i]
                        * (
                            aliasing
                            if periodic_directions[i] or dealias_nonperiodic
                            else 1
                        )
                    )
                    + 1
                )
                for i in range(len(shape))
            )
        else:
            physical_shape = shape
            shape = tuple(
                (
                    math.floor(physical_shape[i] / aliasing)
                    if periodic_directions[i]
                    else physical_shape[i]
                )
                for i in range(len(shape))
            )
        if use_rfftn:
            try:
                rfftn_direction = [
                    i for i in range(len(periodic_directions)) if periodic_directions[i]
                ][-1]
                shape = tuple(
                    (
                        int(shape[i])
                        if i != rfftn_direction
                        else ((shape[i] - 1) // 2) + 1
                    )
                    for i in range(len(shape))
                )
            except IndexError:
                pass
        grid = []
        diff_mats = []
        for dim in range(number_of_dimensions):
            if periodic_directions[dim]:
                if type(scale_factors) == NoneType:
                    scale_factors_.append(2.0 * np.pi)
                grid.append(get_fourier_grid(physical_shape[dim], scale_factors_[dim]))
                diff_mats.append(
                    assemble_fourier_diff_mat(N=physical_shape[dim], order=1)
                    * (2 * np.pi)
                    / scale_factors_[dim]
                )
            else:
                if type(scale_factors) == NoneType:
                    scale_factors_.append(1.0)
                grid.append(get_cheb_grid(physical_shape[dim], scale_factors_[dim]))
                diff_mats.append(assemble_cheb_diff_mat(grid[dim]))
        # make grid immutable
        for gr in grid:
            gr.setflags(write=False)
        mgrid = np.meshgrid(*grid, indexing="ij")
        for mgr in mgrid:
            mgr.setflags(write=False)
        return cls(
            number_of_dimensions=number_of_dimensions,
            periodic_directions=tuple(periodic_directions),
            scale_factors=tuple(scale_factors_),
            shape=tuple(shape),
            grid=tuple(grid),
            diff_mats=tuple(diff_mats),
            mgrid=tuple(mgrid),
            aliasing=aliasing,
            dealias_nonperiodic=dealias_nonperiodic,
        )

    @abstractmethod
    def get_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def get_shape_aliasing(self) -> tuple[int, ...]: ...

    def get_rfftn_direction(self) -> int:
        return self.all_periodic_dimensions()[
            -1
        ]  # rfftn performs the real transform over the last axis

    def number_of_cells(self, direction: int) -> int:
        return len(self.grid[direction])

    def all_dimensions(self) -> Sequence[int]:
        return range(self.number_of_dimensions)
        # return jnp.arange(self.number_of_dimensions)

    def is_periodic(self, direction: int) -> bool:
        return self.periodic_directions[direction]

    def all_periodic_dimensions(self) -> list[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.is_periodic(d)
        ]

    def all_nonperiodic_dimensions(self) -> list[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.is_periodic(d)
        ]

    # @partial(jax.jit, static_argnums=(0,2,3))
    def diff(self, field: "jnp_array", direction: int, order: int = 1) -> "jnp_array":
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

    def enforce_homogeneous_dirichlet(self, mat: "np_float_array") -> "np_float_array":
        """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        homogeneous dirichlet boundary conditions at both ends by setting the
        off-diagonal elements of its first and last rows and columns to zero and
        the diagonal elements to unity."""

        def set_first_mat_row_and_col_to_unit(
            matr: "np_float_array",
        ) -> "np_float_array":
            N = matr.shape[0]
            return np.block(
                [[1, np.zeros((1, N - 1))], [np.zeros((N - 1, 1)), matr[1:, 1:]]]
            )

        def set_last_mat_row_and_col_to_unit(
            matr: "np_float_array",
        ) -> "np_float_array":
            N = matr.shape[0]
            return np.block(
                [[matr[:-1, :-1], np.zeros((N - 1, 1))], [np.zeros((1, N - 1)), 1]]
            )

        return set_last_mat_row_and_col_to_unit(set_first_mat_row_and_col_to_unit(mat))

    def enforce_homogeneous_dirichlet_jnp(self, mat: "np_jnp_array") -> "jnp_array":
        """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        homogeneous dirichlet boundary conditions at both ends by setting the
        off-diagonal elements of its first and last rows and columns to zero and
        the diagonal elements to unity."""

        def set_first_mat_row_and_col_to_unit(
            matr: "jnp_array",
        ) -> "jnp_array":
            N = matr.shape[0]
            return jnp.block(
                [[1, jnp.zeros((1, N - 1))], [jnp.zeros((N - 1, 1)), matr[1:, 1:]]]  # type: ignore[arg-type]
            )

        def set_last_mat_row_and_col_to_unit(
            matr: "jnp_array",
        ) -> "jnp_array":
            N = matr.shape[0]
            return jnp.block(
                [[matr[:-1, :-1], jnp.zeros((N - 1, 1))], [jnp.zeros((1, N - 1)), 1]]  # type: ignore[arg-type]
            )

        return set_last_mat_row_and_col_to_unit(
            set_first_mat_row_and_col_to_unit(jnp.asarray(mat))
        )

    def enforce_inhomogeneous_dirichlet(
        self,
        mat: "np_float_array",
        rhs: "np_jnp_array",
        bc_left: "jsd_float",
        bc_right: "jsd_float",
    ) -> tuple["np_float_array", "np_float_array"]:
        # """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        # inhomogeneous dirichlet boundary conditions at both ends by setting the
        # off-diagonal elements of its first and last rows to zero and
        # the diagonal elements to unity, and the first and last element of the
        # rhs to the desired values bc_left and bc_right."""
        def set_first_mat_row_to_unit(matr: "np_float_array") -> "np_float_array":
            N = matr.shape[0]
            return np.block([[1, np.zeros((1, N - 1))], [matr[1:, :]]])

        def set_last_mat_row_to_unit(matr: "np_float_array") -> "np_float_array":
            N = matr.shape[0]
            return np.block([[matr[:-1, :]], [np.zeros((1, N - 1)), 1]])

        out_mat = set_last_mat_row_to_unit(set_first_mat_row_to_unit(mat))
        out_rhs = np.block([bc_right, rhs[1:-1], bc_left])

        return (out_mat, out_rhs)

    def enforce_inhomogeneous_dirichlet_jnp(
        self,
        mat: "np_jnp_array",
        rhs: "np_jnp_array",
        bc_left: "jsd_float",
        bc_right: "jsd_float",
    ) -> tuple["jnp_array", "jnp_array"]:
        # """Modify a (Chebyshev) differentiation matrix mat in order to fulfill
        # inhomogeneous dirichlet boundary conditions at both ends by setting the
        # off-diagonal elements of its first and last rows to zero and
        # the diagonal elements to unity, and the first and last element of the
        # rhs to the desired values bc_left and bc_right."""
        def set_first_mat_row_to_unit(matr: "jnp_array") -> "jnp_array":
            N = matr.shape[0]
            return jnp.block([[1, jnp.zeros((1, N - 1))], [matr[1:, :]]])  # type: ignore[arg-type]

        def set_last_mat_row_to_unit(matr: "jnp_array") -> "jnp_array":
            N = matr.shape[0]
            return jnp.block([[matr[:-1, :]], [jnp.zeros((1, N - 1)), 1]])  # type: ignore[arg-type]

        out_mat = set_last_mat_row_to_unit(set_first_mat_row_to_unit(jnp.asarray(mat)))
        out_rhs = jnp.block([bc_right, rhs[1:-1], bc_left])

        return (out_mat, out_rhs)

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction: int) -> "np_float_array":
        """Assemble the Chebyshev differentiation matrix of second order with
        homogeneous Dirichlet boundary conditions enforced by setting the first
        and last rows and columns to one (diagonal elements) and zero
        (off-diagonal elements)"""

        return self.enforce_homogeneous_dirichlet(
            np.linalg.matrix_power(self.diff_mats[direction], 2)
        )

    def integrate(
        self,
        field: "jnp_array",
        direction: int,
        order: int = 1,
        bc_left: Optional[float] = None,
        bc_right: Optional[float] = None,
    ) -> "jnp_array":
        """Calculate the integral or order for field in direction subject to the
        boundary conditions bc_left and/or bc_right. Since this is difficult to
        generalize, only cases that are needed are implemented."""

        def safe_is_nonzero(input: Optional[float]) -> bool:
            if type(input) == NoneType:
                return False
            else:
                assert input is not None
                return abs(input) > 1e-20

        if (safe_is_nonzero(bc_left)) or (safe_is_nonzero(bc_right)):
            raise Exception("Only homogeneous dirichlet conditions currently supported")

        assert order <= 2, "Integration only supported up to second order"

        def set_first_mat_row_and_col_to_unit(matr: "jnp_array") -> "jnp_array":
            if bc_right == None:
                return matr
            N = matr.shape[0]
            out = jnp.block(
                [
                    ([jnp.ones((1)), jnp.zeros((1, N - 1))]),
                    ([jnp.zeros((N - 1, 1)), matr[1:, 1:]]),
                ]
            )
            return out

        def set_last_mat_row_and_col_to_unit(matr: "jnp_array") -> "jnp_array":
            if bc_left == None:
                return matr
            N = matr.shape[0]
            out = jnp.block(
                [
                    ([matr[:-1, :-1], jnp.zeros((N - 1, 1))]),
                    ([jnp.zeros((1, N - 1)), jnp.ones((1))]),
                ]
            )
            return out

        def set_first_of_field(
            field: "jnp_array", new_first: Union["jnp_array", float]
        ) -> "jnp_array":
            N = field.shape[direction]
            inds = jnp.arange(1, N)
            inner = field.take(indices=inds, axis=direction)
            out = jnp.pad(
                inner,
                [(0, 0) if d != direction else (1, 0) for d in self.all_dimensions()],
                mode="constant",
                constant_values=new_first,
            )
            return out

        def set_last_of_field(
            field: "jnp_array", new_last: Union["jnp_array", float]
        ) -> "jnp_array":
            N = field.shape[direction]
            inds = jnp.arange(0, N - 1)
            inner = field.take(indices=inds, axis=direction)
            out = jnp.pad(
                inner,
                [(0, 0) if d != direction else (0, 1) for d in self.all_dimensions()],
                mode="constant",
                constant_values=new_last,
            )
            return out

        def set_first_and_last_of_field(
            field: "jnp_array",
            first: Union["jnp_array", float],
            last: Union["jnp_array", float],
        ) -> "jnp_array":
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
                    assert bc_right is not None
                    mat = set_first_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                    b = set_first_of_field(field, bc_right)
                elif type(bc_left) != NoneType and type(bc_right) == NoneType:
                    assert bc_left is not None
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

    def update_boundary_conditions(self, field: "jnp_array") -> "jnp_array":
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

    def curl(self, field: "jnp_array") -> "jnp_array":
        """Compute the curl of field."""
        # assert len(field) == 3, "rotation only defined in 3 dimensions"
        u_y = self.diff(field[0, ...], 1)
        u_z = self.diff(field[0, ...], 2)
        v_x = self.diff(field[1, ...], 0)
        v_z = self.diff(field[1, ...], 2)
        w_x = self.diff(field[2, ...], 0)
        w_y = self.diff(field[2, ...], 1)

        curl_0 = w_y - v_z
        curl_1 = u_z - w_x
        curl_2 = v_x - u_y

        return jnp.array([curl_0, curl_1, curl_2])

    def cross_product(self, field_1: "jnp_array", field_2: "jnp_array") -> "jnp_array":
        """Compute the cross (or vector) product of field_1 and field_2."""
        out_0 = field_1[1, ...] * field_2[2, ...] - field_1[2, ...] * field_2[1, ...]
        out_1 = field_1[2, ...] * field_2[0, ...] - field_1[0, ...] * field_2[2, ...]
        out_2 = field_1[0, ...] * field_2[1, ...] - field_1[1, ...] * field_2[0, ...]
        return jnp.array([out_0, out_1, out_2])

    def laplacian(self, field: "jnp_array") -> "jnp_array":
        # TODO generalize dimensions
        return self.diff(field, 0, 2) + self.diff(field, 1, 2) + self.diff(field, 2, 2)

    def divergence(self, field: "jnp_array") -> "jnp_array":
        # TODO generalize dimensions
        return self.diff(field[0], 0) + self.diff(field[1], 1) + self.diff(field[2], 2)


# @dataclasses.dataclass(frozen=True, kw_only=True)
@dataclasses.dataclass(frozen=True)
class PhysicalDomain(Domain):
    """Domain that lives in physical space (as opposed to Fourier space)."""

    def __hash__(self) -> int:
        return hash(
            (
                self.number_of_dimensions,
                self.periodic_directions,
                self.scale_factors,
                self.shape,
                self.aliasing,
                self.dealias_nonperiodic,
            )
        )

    def __eq__(self, other: Any) -> bool:
        return hash(self) is hash(other)

    def get_shape(self) -> tuple[int, ...]:
        return self.shape

    def get_shape_aliasing(self) -> tuple[int, ...]:
        return tuple(map(lambda x: len(x), self.grid))

    def get_extent(self, i: int) -> float:
        return self.scale_factors[i] * 1.0 if self.is_periodic(i) else 2.0

    def hat(self) -> "FourierDomain":
        """Create a Fourier transform of the present domain in all periodic
        directions and return the resulting domain."""

        return FourierDomain.FromPhysicalDomain(self)

    def field_hat(self, field: "jnp_array") -> "jnp_array":
        """Compute the Fourier transform of field."""
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        out = rfftn_jit(field, tuple(self.all_periodic_dimensions())) / scaling_factor

        Ns = [self.number_of_cells(i) for i in self.all_dimensions()]
        ks = [
            int((Ns[i] - Ns[i] * (1 - 1 / self.aliasing)) / 2)
            for i in self.all_dimensions()
        ]

        # TODO rewrite this to improve GPU performance
        periodic_dims = (
            self.all_periodic_dimensions()[:-1]
            if use_rfftn
            else self.all_periodic_dimensions()
        )
        for i in periodic_dims:
            out_1 = out.take(indices=jnp.arange(0, ks[i]), axis=i)
            out_2 = out.take(indices=jnp.array([ks[i]]), axis=i)
            out_3 = out.take(indices=jnp.arange(Ns[i] - ks[i] + 1, Ns[i]), axis=i)
            out = jnp.concatenate([out_1, out_2, jnp.conjugate(out_2), out_3], axis=i)
        if use_rfftn:
            i = self.all_periodic_dimensions()[-1]
            out = out.take(indices=jnp.arange(0, ks[i] + 1), axis=i)

        if self.dealias_nonperiodic:
            out_ = jnp.zeros(self.get_shape())
            for i in self.all_nonperiodic_dimensions():
                out_grid = get_cheb_grid(self.get_shape()[i])
                other_dims = [j for j in self.all_dimensions() if j != i]
                other_shape = tuple([self.get_shape()[i] for i in other_dims])
                for N in np.ndindex(other_shape):
                    index = tuple(
                        [
                            N[other_dims.index(i)] if i in other_dims else slice(None)
                            for i in self.all_dimensions()
                        ]
                    )
                    cheb_coeffs = chebyshev.chebfit(
                        self.grid[i], out[index], self.get_shape_aliasing()[i]
                    )
                    cheb_coeffs = cheb_coeffs[: self.get_shape()[i]]
                    out_ = out_.at[index].set(chebyshev.chebval(out_grid, cheb_coeffs))
            out = out_

        return cast("jnp_array", out.astype(jnp.complex128))


# @dataclasses.dataclass(frozen=True, init=True, kw_only=True)
@dataclasses.dataclass(frozen=True, init=True)
class FourierDomain(Domain):
    """Same as PhysicalDomain but lives in Fourier space."""

    physical_domain: Optional[PhysicalDomain] = (
        None  # the physical domain it is based on
    )

    def __hash__(self) -> int:
        return hash(
            (
                self.number_of_dimensions,
                self.periodic_directions,
                self.scale_factors,
                self.shape,
                self.aliasing,
                self.dealias_nonperiodic,
            )
        )

    def __eq__(self, other: Any) -> bool:
        return hash(self) is hash(other)

    @classmethod
    def FromPhysicalDomain(cls, physical_domain: PhysicalDomain) -> "FourierDomain":
        """Create a Fourier transform of the present domain in all periodic
        directions and return the resulting domain."""

        def fftshift(inp: "np_float_array", i: int) -> "np_float_array":
            if physical_domain.periodic_directions[i]:
                rfftn_direction = physical_domain.get_rfftn_direction()
                N = len(inp)
                if use_rfftn and i == rfftn_direction:
                    return inp * (2 * np.pi) / physical_domain.scale_factors[i]
                else:
                    return (
                        (np.block([inp[N // 2 :], inp[: N // 2]]) - N // 2)
                        * (2 * np.pi)
                        / physical_domain.scale_factors[i]
                    )
            else:
                return inp

        Ns = []
        for i in physical_domain.all_dimensions():
            Ns.append(physical_domain.shape[i])
        fourier_grid = []
        for i in physical_domain.all_dimensions():
            if (
                physical_domain.periodic_directions[i]
                or physical_domain.dealias_nonperiodic
            ):
                fourier_grid.append(np.linspace(0, Ns[i] - 1, int(Ns[i])))
            else:
                fourier_grid.append(physical_domain.grid[i])
        fourier_grid_shifted = list(
            map(fftshift, fourier_grid, physical_domain.all_dimensions())
        )
        grid = fourier_grid_shifted
        mgrid = np.meshgrid(*fourier_grid_shifted, indexing="ij")
        diff_mats = [
            (
                physical_domain.diff_mats[i]
                if physical_domain.is_periodic(i)
                else assemble_cheb_diff_mat(grid[i])
            )
            for i in physical_domain.all_dimensions()
        ]
        out = FourierDomain(
            number_of_dimensions=physical_domain.number_of_dimensions,
            periodic_directions=tuple(physical_domain.periodic_directions),
            scale_factors=tuple(physical_domain.scale_factors),
            shape=tuple(physical_domain.shape),
            grid=tuple(grid),
            diff_mats=tuple(diff_mats),
            mgrid=tuple(mgrid),
            aliasing=physical_domain.aliasing,
            dealias_nonperiodic=physical_domain.dealias_nonperiodic,
            physical_domain=physical_domain,
        )
        return out

    def get_shape(self) -> tuple[int, ...]:
        return self.shape

    def get_shape_aliasing(self) -> tuple[int, ...]:
        assert self.physical_domain is not None
        return tuple(map(lambda x: len(x), self.physical_domain.grid))

    def assemble_poisson_matrix(self) -> "np_complex_array":
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
        eye_bc = np.block(
            [
                [np.zeros((bc_padding, n))],
                [
                    np.zeros((n - 2 * bc_padding, bc_padding)),
                    np.eye(n - 2 * bc_padding),
                    np.zeros((n - 2 * bc_padding, bc_padding)),
                ],
                [np.zeros((bc_padding, n))],
            ]
        )
        k1 = self.grid[self.all_periodic_dimensions()[0]]
        k2 = self.grid[self.all_periodic_dimensions()[1]]
        k1sq = k1**2
        k2sq = k2**2
        mat = np.array(
            [
                [np.linalg.inv((-(k1sq_ + k2sq_)) * eye_bc + y_mat) for k2sq_ in k2sq]
                for k1sq_ in k1sq
            ]
        )
        return mat

    def diff(
        self,
        field_hat: "jnp_array",
        direction: int,
        order: int = 1,
    ) -> "jnp_array":
        """Calculate and return the derivative of given order for field in
        direction."""
        if direction in self.all_periodic_dimensions():
            diff_array = (1j * np.array(self.mgrid[direction])) ** order
            f_diff: "jnp_array" = jnp.array(diff_array * field_hat)
        else:
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
                ind, field_hat, np.linalg.matrix_power(self.diff_mats[direction], order)
            )
        return f_diff

    def curl(self, field_hat: "jnp_array") -> "jnp_array":
        """Compute the curl of field."""
        u_y = self.diff(field_hat[0, ...], 1)
        u_z = self.diff(field_hat[0, ...], 2)
        v_x = self.diff(field_hat[1, ...], 0)
        v_z = self.diff(field_hat[1, ...], 2)
        w_x = self.diff(field_hat[2, ...], 0)
        w_y = self.diff(field_hat[2, ...], 1)

        curl_0 = w_y - v_z
        curl_1 = u_z - w_x
        curl_2 = v_x - u_y

        return jnp.array([curl_0, curl_1, curl_2])

    def project_onto_domain(
        self,
        target_domain: PhysicalDomain,
        field_hat: "jnp_array",
    ) -> "jnp_array":
        initial_shape = self.shape
        target_domain_hat = target_domain.hat()
        target_shape = target_domain_hat.shape

        for i in self.all_periodic_dimensions():
            N = initial_shape[i]
            N_target = target_shape[i]
            if N > N_target:
                data_1 = field_hat.take(
                    indices=jnp.arange(0, (N_target - 1) // 2 + 1), axis=i
                )
                data_2 = field_hat.take(
                    indices=jnp.arange(N - (N_target - 1) // 2, N), axis=i
                )
                field_hat = jnp.concatenate([data_1, data_2], axis=i)
            elif N < N_target:
                zeros_shape = [
                    field_hat.shape[dim] if dim != i else N_target - N
                    for dim in self.all_dimensions()
                ]
                extra_zeros = jnp.zeros(zeros_shape)
                data_1 = field_hat.take(indices=jnp.arange(0, (N - 1) // 2 + 1), axis=i)
                data_2 = field_hat.take(indices=jnp.arange((N - 1) // 2 + 1, N), axis=i)
                field_hat = jnp.concatenate([data_1, extra_zeros, data_2], axis=i)
            else:
                pass

        field_ = jnp.zeros(target_domain.get_shape())
        for i in self.all_nonperiodic_dimensions():
            N = initial_shape[i]
            N_target = target_shape[i]
            if N != N_target:
                out_grid = get_cheb_grid(N_target)
                in_grid = get_cheb_grid(self.get_shape()[i])
                other_dims = [j for j in self.all_dimensions() if j != i]
                other_shape = tuple([self.get_shape()[i] for i in other_dims])

                for ns in np.ndindex(other_shape):
                    index = tuple(
                        [
                            ns[other_dims.index(i)] if i in other_dims else slice(None)
                            for i in self.all_dimensions()
                        ]
                    )
                    cheb_coeffs = chebyshev.chebfit(
                        in_grid, field_hat[index], self.get_shape()[i]
                    )
                    if N > N_target:
                        cheb_coeffs = cheb_coeffs[:N_target]
                    else:
                        extra_zeros = jnp.zeros((len(out_grid) - len(self.grid[i])))
                        cheb_coeffs = jnp.concatenate([cheb_coeffs, extra_zeros])
                    field_ = field_.at[index].set(
                        chebyshev.chebval(out_grid, cheb_coeffs)
                    )
                field_hat = field_
            else:
                pass  # the shape is already correct (N = N_target), no need to do anything

        return field_hat.astype(jnp.complex128)

    def project_onto_domain_nonfourier(
        self,
        target_domain: PhysicalDomain,
        field: "jnp_array",
    ) -> "jnp_array":
        initial_shape = self.shape
        target_domain_hat = target_domain.hat()
        target_shape = target_domain_hat.shape

        field_ = jnp.zeros(target_shape)
        for i in self.all_nonperiodic_dimensions():
            N = initial_shape[i]
            N_target = target_shape[i]
            if N != N_target:
                out_grid = get_cheb_grid(N_target)
                in_grid = get_cheb_grid(self.get_shape()[i])
                other_dims = [j for j in self.all_dimensions() if j != i]
                other_shape = tuple([self.get_shape()[i] for i in other_dims])

                for ns in np.ndindex(other_shape):
                    index = tuple(
                        [
                            ns[other_dims.index(i)] if i in other_dims else slice(None)
                            for i in self.all_dimensions()
                        ]
                    )
                    cheb_coeffs = chebyshev.chebfit(
                        in_grid, field[index], self.get_shape()[i]
                    )
                    if N > N_target:
                        cheb_coeffs = cheb_coeffs[:N_target]
                    else:
                        extra_zeros = jnp.zeros((len(out_grid) - len(self.grid[i])))
                        cheb_coeffs = jnp.concatenate([cheb_coeffs, extra_zeros])
                    field_ = field_.at[index].set(
                        chebyshev.chebval(out_grid, cheb_coeffs)
                    )
                field = field_
            else:
                pass  # the shape is already correct (N = N_target), no need to do anything

        return field.astype(jnp.float64)

    def filter_field(self, field_hat: "jnp_array") -> "jnp_array":
        if self.dealias_nonperiodic:
            N_coarse = tuple(
                self.shape[i] - int(self.shape[i] * (1 - 1 / self.aliasing))
                for i in self.all_dimensions()
            )
            N_coarse = tuple(
                (
                    N_coarse[i]
                    if N_coarse[i] % 2 == 0 or not self.periodic_directions[i]
                    else N_coarse[i] + 1
                )
                for i in self.all_dimensions()
            )
            coarse_domain = PhysicalDomain.create(
                N_coarse, self.periodic_directions, self.scale_factors, 1
            )
            coarse_domain_hat = coarse_domain.hat()

            coarse_field_hat = self.project_onto_domain(coarse_domain, field_hat)
            assert self.physical_domain is not None
            fine_field_hat = coarse_domain_hat.project_onto_domain(
                self.physical_domain, coarse_field_hat
            )

            return fine_field_hat
        else:
            return self.filter_field_fourier_only(field_hat)

    def filter_field_fourier_only(self, field_hat: "jnp_array") -> "jnp_array":
        N_coarse = tuple(
            self.shape[i]
            - (
                # int(self.shape[i] * (1 - 1 / self.aliasing))
                3
                if self.is_periodic(i)
                else 0
            )
            for i in self.all_dimensions()
        )
        N_coarse = tuple(
            (
                N_coarse[i]
                if N_coarse[i] % 2 == 0 or not self.periodic_directions[i]
                else N_coarse[i] + 1
            )
            for i in self.all_dimensions()
        )
        coarse_domain = PhysicalDomain.create(
            N_coarse, self.periodic_directions, self.scale_factors, 1
        )
        coarse_domain_hat = coarse_domain.hat()

        coarse_field_hat = self.project_onto_domain(coarse_domain, field_hat)
        assert self.physical_domain is not None
        fine_field_hat = coarse_domain_hat.project_onto_domain(
            self.physical_domain, coarse_field_hat
        )

        return fine_field_hat

    def filter_field_nonfourier_only(self, field: "jnp_array") -> "jnp_array":
        N_coarse = tuple(
            self.shape[i]
            - (
                int(self.shape[i] * (1 - 1 / self.aliasing))
                if not self.is_periodic(i)
                else 0
            )
            for i in self.all_dimensions()
        )
        coarse_domain = PhysicalDomain.create(
            N_coarse, self.periodic_directions, self.scale_factors, 1
        )
        coarse_domain_hat = coarse_domain.hat()

        coarse_field = self.project_onto_domain_nonfourier(coarse_domain, field)
        assert self.physical_domain is not None
        fine_field = coarse_domain_hat.project_onto_domain_nonfourier(
            self.physical_domain, coarse_field
        )

        return fine_field

    def solve_poisson_fourier_field_slice(
        self,
        field: "jnp_array",
        mat: "np_jnp_array",
        k1: Optional[int],
        k2: Optional[int],
    ) -> "jnp_array":
        """Solve the poisson equation with field as the right-hand side for a
        one-dimensional slice at the wavenumbers k1 and k2. Use the provided
        differentiation matrix mat."""
        if type(k1) == NoneType and type(k2) == NoneType:
            mat_inv = mat[:, :]
        else:
            assert k1 is not None
            assert k2 is not None
            mat_inv = mat[k1, k2, :, :]
        rhs_hat = field
        out_field = jnp.array(mat_inv @ rhs_hat)
        return out_field

    def update_boundary_conditions_fourier_field_slice(
        self, field: "jnp_array", non_periodic_direction: int
    ) -> "jnp_array":
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

    def field_no_hat(self, field_hat: "jnp_array") -> "jnp_array":
        """Compute the inverse Fourier transform of field."""
        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.scale_factors[i] / (2 * jnp.pi)

        Ns = [int(self.number_of_cells(i)) for i in self.all_dimensions()]
        ks = [int((Ns[i]) / 2) for i in self.all_dimensions()]
        # TODO rewrite this to improve GPU performance
        periodic_dims = (
            self.all_periodic_dimensions()[:-1]
            if use_rfftn
            else self.all_periodic_dimensions()
        )
        for i in periodic_dims:
            field_1 = field_hat.take(indices=jnp.arange(0, ks[i] + 1), axis=i)
            zeros_shape = [
                (
                    field_1.shape[dim]
                    if dim != i
                    else math.ceil(Ns[i] * (self.aliasing - 1))
                )
                for dim in self.all_dimensions()
            ]
            field_2 = field_hat.take(indices=jnp.arange(Ns[i] - ks[i], Ns[i]), axis=i)
            if zeros_shape[i] == 0:
                field_2 = field_2.take(indices=jnp.arange(1, field_2.shape[i]), axis=i)
            extra_zeros = jnp.zeros(zeros_shape)
            field_hat = jnp.concatenate([field_1, extra_zeros, field_2], axis=i)
        if use_rfftn:
            i = self.all_periodic_dimensions()[-1]
            zeros_shape = [
                (
                    field_hat.shape[dim]
                    if dim != i
                    else math.ceil(Ns[i] * (self.aliasing - 1))
                )
                for dim in self.all_dimensions()
            ]
            extra_zeros = jnp.zeros(zeros_shape)
            field_hat = jnp.concatenate([field_hat, extra_zeros], axis=i)

        if self.dealias_nonperiodic:
            assert self.physical_domain is not None
            out_grid = self.physical_domain.grid
            field_ = jnp.zeros(self.get_shape_aliasing())
            for i in self.all_nonperiodic_dimensions():
                in_grid = get_cheb_grid(self.get_shape()[i])
                other_dims = [j for j in self.all_dimensions() if j != i]
                other_shape = tuple([self.get_shape()[i] for i in other_dims])
                for N in np.ndindex(other_shape):
                    index = tuple(
                        [
                            N[other_dims.index(i)] if i in other_dims else slice(None)
                            for i in self.all_dimensions()
                        ]
                    )
                    cheb_coeffs = chebyshev.chebfit(
                        in_grid, field_hat[index], self.get_shape()[i]
                    )
                    extra_zeros = jnp.zeros((len(out_grid[i]) - len(self.grid[i])))
                    cheb_coeffs = jnp.concatenate([cheb_coeffs, extra_zeros])
                    field_ = field_.at[index].set(
                        chebyshev.chebval(out_grid[i], cheb_coeffs)
                    )
                field_hat = field_

        out = cast(
            "jnp_array",
            irfftn_jit(field_hat, tuple(self.all_periodic_dimensions())).real
            / (1 / scaling_factor),
        )
        return out

    def diff_fourier_field_slice(
        self,
        field: "jnp_array",
        orientation: int,
        direction: int,
        order: int = 1,
        k: int = 0,
    ) -> "jnp_array":
        """Calculate and return the derivative of given order for a Fourier
        field slice in direction."""
        if orientation == direction:
            return jnp.array(
                np.linalg.matrix_power(self.diff_mats[direction], order) @ field
            )
        else:
            j_k = 1j * self.grid[direction][k]
            return cast("jnp_array", j_k**order * field)

    def integrate_fourier_field_slice(
        self,
        field: "jnp_array",
        orientation: int,
        direction: int,
        order: int = 1,
        bc_left: Optional[float] = None,
        bc_right: Optional[float] = None,
    ) -> "jnp_array":
        """Calculate and return the derivative of given order for a Fourier
        field slice in direction."""

        def safe_is_nonzero(input: Optional[float]) -> bool:
            if type(input) == NoneType:
                return False
            else:
                assert input is not None
                return abs(input) > 1e-20

        if (safe_is_nonzero(bc_left)) or (safe_is_nonzero(bc_right)):
            raise Exception("Only homogeneous dirichlet conditions currently supported")

        assert order <= 2, "Integration only supported up to second order"

        def set_first_mat_row_and_col_to_unit(matr: "jnp_array") -> "jnp_array":
            if bc_right == None:
                return matr
            N = matr.shape[0]
            out = jnp.block(
                [
                    ([jnp.ones((1)), jnp.zeros((1, N - 1))]),
                    ([jnp.zeros((N - 1, 1)), matr[1:, 1:]]),
                ]
            )
            return out

        def set_last_mat_row_and_col_to_unit(matr: "jnp_array") -> "jnp_array":
            if bc_left == None:
                return matr
            N = matr.shape[0]
            out = jnp.block(
                [
                    ([matr[:-1, :-1], jnp.zeros((N - 1, 1))]),
                    ([jnp.zeros((1, N - 1)), jnp.ones((1))]),
                ]
            )
            return out

        if orientation == direction:
            if order == 1:
                if type(bc_right) is not NoneType and type(bc_left) is NoneType:
                    assert bc_right is not None
                    mat = set_first_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                    # b = set_first_of_field(field, bc_right)
                elif type(bc_left) is not NoneType and type(bc_right) is NoneType:
                    assert bc_left is not None
                    mat = set_last_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                    # b = set_last_of_field(field, bc_left)

                else:
                    mat = set_last_mat_row_and_col_to_unit(
                        set_first_mat_row_and_col_to_unit(
                            jnp.linalg.matrix_power(self.diff_mats[direction], order)
                        )
                    )
            elif order == 2:
                mat = set_last_mat_row_and_col_to_unit(
                    set_first_mat_row_and_col_to_unit(
                        jnp.linalg.matrix_power(self.diff_mats[direction], order)
                    )
                )
                # b_right = 0.0
                # b_left = 0.0
                # b = set_first_and_last_of_field(field, b_right, b_left)

        else:
            raise Exception(
                "Integration not implemented in periodic directions, use Fourier integration instead."
            )

        inv_mat = jnp.linalg.inv(mat)

        out: "jnp_array" = inv_mat @ field

        return out

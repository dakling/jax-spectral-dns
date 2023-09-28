#!/usr/bin/env python3

from types import NoneType
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np


class Field:
    plotting_dir = "./plots/"

    def __init__(self, domain, field, name="field"):
        self.domain = domain
        self.field = field
        self.name = name
        self.field_hat = None

    @classmethod
    def FromFunc(cls, domain, func=None, name="field"):
        if not func:
            func = lambda x: 0.0 * math.prod(x)
        field = jnp.array(list(map(lambda *x: func(x), *domain.mgrid)))
        return cls(domain, field, name)

    @classmethod
    def FromRandom(cls, domain, seed=0, interval=(-0.1, 0.1), name="field"):
        # TODO generate "nice" random fields
        key = jax.random.PRNGKey(seed)
        zero_field = Field.FromFunc(domain)
        rands = []
        for i in range(zero_field.number_of_dofs()):
            key, subkey = jax.random.split(key)
            rands.append(
                jax.random.uniform(subkey, minval=interval[0], maxval=interval[1])
            )
        field = jnp.array(rands).reshape(zero_field.domain.shape)
        return cls(domain, field, name)
        pass

    def __repr__(self):
        # fig, ax = plt.subplots(1,1)
        # ax.plot(self.mgrid[0], self.field)
        self.plot()
        return str(self.field)

    def __getitem__(self, index):
        return self.field[index]

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        if other.name[0] == "-":
            new_name = self.name + " - " + other.name[1:]
        else:
            new_name = self.name + " + " + other.name
        return Field(self.domain, self.field + other.field, name=new_name)

    def __sub__(self, other):
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            try:
                new_name = self.name + " * " + other.name
            except Exception:
                new_name = "field"
            return Field(self.domain, self.field * other.field, name=new_name)
        else:
            try:
                if other.real >= 0:
                    new_name = str(other) + self.name
                elif other == 1:
                    new_name = self.name
                elif other == -1:
                    new_name = "-" + self.name
                else:
                    new_name = "(" + str(other) + ") " + self.name
            except Exception:
                new_name = "field"
            return Field(self.domain, self.field * other, name=new_name)

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other):
        if type(other) == Field:
            raise Exception("Don't know how to divide by another field")
        else:
            try:
                if other.real >= 0:
                    new_name = self.name + "/" + other
                elif other == 1:
                    new_name = self.name
                elif other == -1:
                    new_name = "-" + self.name
                else:
                    new_name = self.name + "/ (" + str(other) + ") "
            except Exception:
                new_name = "field"
            return Field(self.domain, self.field * other, name=new_name)

    def __abs__(self):
        return (
            jnp.linalg.norm(self.field) / self.number_of_dofs()
        )  # TODO use integration or something more sophisticated

    def number_of_dimensions(self):
        return len(self.all_dimensions())

    def number_of_dofs(self):
        return int(math.prod(self.domain.shape))

    def plot_center(self, dimension, *other_fields):
        if self.domain.number_of_dimensions == 1:
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.domain.grid[0], self.field, label=self.name)
            for other_field in other_fields:
                ax.plot(
                    self.domain.grid[dimension],
                    other_field.field,
                    label=other_field.name,
                )
            fig.legend()
            fig.savefig(self.plotting_dir + "plot_cl_" + self.name + ".pdf")
        elif self.domain.number_of_dimensions == 2:
            fig, ax = plt.subplots(1, 1)
            other_dim = [i for i in self.all_dimensions() if i != dimension][0]
            N_c = len(self.domain.grid[other_dim]) // 2
            ax.plot(
                self.domain.grid[dimension],
                self.field.take(indices=N_c, axis=other_dim),
                label=self.name,
            )
            for other_field in other_fields:
                ax.plot(
                    self.domain.grid[dimension],
                    other_field.field.take(indices=N_c, axis=other_dim),
                    label=other_field.name,
                )
            fig.legend()
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_"
                + ["x", "y"][dimension]
                + ".pdf"
            )
        elif self.domain.number_of_dimensions == 3:
            fig, ax = plt.subplots(1, 1)
            other_dim = [i for i in self.all_dimensions() if i != dimension]
            N_c = [len(self.domain.grid[dim]) // 2 for dim in other_dim]
            ax.plot(
                self.domain.grid[dimension],
                self.field.take(indices=N_c[1], axis=other_dim[1]).take(
                    indices=N_c[0], axis=other_dim[0]
                ),
                label=self.name,
            )
            for other_field in other_fields:
                ax.plot(
                    self.domain.grid[dimension],
                    other_field.field.take(indices=N_c[1], axis=other_dim[1]).take(
                        indices=N_c[0], axis=other_dim[0]
                    ),
                    label=other_field.name,
                )
            fig.legend()
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_"
                + ["x", "y", "z"][dimension]
                + ".pdf"
            )
        else:
            raise Exception("Not implemented yet")

    def plot(self, *other_fields):
        if self.domain.number_of_dimensions == 1:
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                self.domain.grid[0],
                self.field,
                label=self.name,
            )
            for other_field in other_fields:
                ax.plot(
                    self.domain.grid[0],
                    other_field.field,
                    label=other_field.name,
                )
            fig.legend()
            fig.savefig(self.plotting_dir + "plot_" + self.name + ".pdf")

        elif self.domain.number_of_dimensions == 2:
            fig = plt.figure(figsize=(15, 5))
            ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)]
            ax3d = fig.add_subplot(1, 3, 3, projection="3d")
            for dimension in self.all_dimensions():
                other_dim = [i for i in self.all_dimensions() if i != dimension][0]
                N_c = len(self.domain.grid[other_dim]) // 2
                ax[dimension].plot(
                    self.domain.grid[dimension],
                    self.field.take(indices=N_c, axis=other_dim),
                    label=self.name,
                )
                ax3d.plot_surface(
                    self.domain.mgrid[0], (self.domain.mgrid[1]), self.field
                )
                for other_field in other_fields:
                    ax[dimension].plot(
                        self.domain.grid[dimension],
                        other_field.field.take(indices=N_c, axis=other_dim),
                        label=other_field.name,
                    )
                    ax3d.plot_surface(
                        self.domain.mgrid[0], (self.domain.mgrid[1]), other_field.field
                    )
                fig.legend()
                fig.savefig(self.plotting_dir + "plot_" + self.name + ".pdf")
        elif self.domain.number_of_dimensions == 3:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for dimension in self.all_dimensions():
                other_dim = [i for i in self.all_dimensions() if i != dimension]
                N_c = [len(self.domain.grid[dim]) // 2 for dim in other_dim]
                ax[dimension].plot(
                    self.domain.grid[dimension],
                    self.field.take(indices=N_c[1], axis=other_dim[1]).take(
                        indices=N_c[0], axis=other_dim[0]
                    ),
                    label=self.name,
                )
                for other_field in other_fields:
                    ax[dimension].plot(
                        self.domain.grid[dimension],
                        other_field.field.take(indices=N_c[1], axis=other_dim[1]).take(
                            indices=N_c[0], axis=other_dim[0]
                        ),
                        label=other_field.name,
                    )
            fig.legend()
            fig.savefig(self.plotting_dir + "plot_" + self.name + ".pdf")
        else:
            raise Exception("Not implemented yet")

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

    def pad_mat_with_zeros(self):
        return jnp.block(
            [
                [jnp.zeros((1, self.field.shape[1] + 2))],
                [
                    jnp.zeros((self.field.shape[0], 1)),
                    self.field,
                    jnp.zeros((self.field.shape[0], 1)),
                ],
                [jnp.zeros((1, self.field.shape[1] + 2))],
            ]
        )

    def update_boundary_conditions(self):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        for dim in self.all_nonperiodic_dimensions():
            self.field = jnp.take(
                self.field,
                jnp.array(list(range(len(self.domain.grid[dim]))))[1:-1],
                axis=dim,
            )
            self.field = jnp.pad(
                self.field,
                [
                    (0, 0) if self.domain.periodic_directions[d] else (1, 1)
                    for d in self.all_dimensions()
                ],
                mode="constant",
                constant_values=0.0,
            )

    def hat(self):
        # self.field_hat = FourierField.FromField(self)
        return FourierField.FromField(self)

    def diff(self, direction, order=1):
        name_suffix = "".join([["x", "y", "z"][direction] for _ in range(order)])
        return Field(
            self.domain,
            self.domain.diff(self.field, direction, order),
            self.name + "_" + name_suffix,
        )

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction):
        return self.domain.get_cheb_mat_2_homogeneous_dirichlet(direction)

    def integrate(self, direction, order=1, bc_left=None, bc_right=None):
        out_bc = self.domain.integrate(self.field, direction, order, bc_left, bc_right)
        return Field(self.domain, out_bc, name=self.name + "_int")

    def nabla(self):
        out = [self.diff(0)]
        out[0].name = "nabla_" + self.name + "_" + str(0)
        for dim in self.all_dimensions()[1:]:
            out.append(self.diff(dim))
            out[dim].name = "nabla_" + self.name + "_" + str(dim)
        return VectorField(out)

    def laplacian(self):
        out = self.diff(0, 2)
        for dim in self.all_dimensions()[1:]:
            out += self.diff(dim, 2)
        out.name = "lap_" + self.name
        return out

    def solve_poisson(self, rhs):
        rhs_hat = rhs.hat()
        denom = 0.0
        for direction in self.all_periodic_dimensions():
            mgrid = self.domain.mgrid[direction]
            for i in reversed(self.all_periodic_dimensions()):
                N = mgrid.shape[i]
                inds = jnp.array(list(range(1, N)))
                mgrid = mgrid.take(indices=inds, axis=i)

            denom += (1j * mgrid) ** 2
        out_0 = 0.0
        field = rhs_hat.field
        for i in reversed(self.all_periodic_dimensions()):
            N = field.shape[i]
            inds = jnp.array(list(range(1, N)))
            field = field.take(indices=inds, axis=i)
        out_field = jnp.pad(
            field / denom,
            [(1, 0) for _ in self.all_periodic_dimensions()],
            mode="constant",
            constant_values=out_0,
        )
        out_fourier = FourierField(self.domain, out_field, name=self.name + "_poisson")
        self.field = out_fourier.no_hat().field
        return out_fourier.no_hat()

    def perform_explicit_euler_step(self, eq, dt, i):
        new_u = self + eq * dt
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        return new_u

    def perform_time_step(self, eq, dt, i):
        return self.perform_explicit_euler_step(eq, dt, i)


class VectorField:
    def __init__(self, elements, name="vector_field"):
        self.elements = elements
        self.name = name

    def __getattr__(self, attr):
        def on_all(*args, **kwargs):
            acc = []
            for obj in self.elements:
                acc += [getattr(obj, attr)(*args, **kwargs)]
            return acc

        return on_all

    def __getitem__(self, index):
        return self.elements[index]

    def __iter__(self):
        return iter(self.elements)

    def append(self, element):
        return self.elements.append(element)

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        out = ""
        for elem in self.elements:
            out += str(elem)
        return out

    def cross_product(self, other):
        out_0 = self[2] * other[1] - self[1] * other[2]
        out_1 = self[0] * other[2] - self[2] * other[0]
        out_2 = self[1] * other[0] - self[0] * other[1]
        return VectorField([out_0, out_1, out_2])

    def curl(self):
        assert len(self) == 3, "rotation only defined in 3 dimensions"
        for f in self:
            assert (
                f.number_of_dimensions() == 3
            ), "rotation only defined in 3 dimensions"
        u_y = self[0].diff(1)
        u_z = self[0].diff(2)
        v_x = self[1].diff(0)
        v_z = self[1].diff(2)
        w_x = self[2].diff(0)
        w_y = self[2].diff(1)

        curl_0 = w_y - v_z
        curl_1 = u_z - w_x
        curl_2 = v_x - u_y

        return VectorField([curl_0, curl_1, curl_2])


class FourierField(Field):
    def __init__(self, domain, field, name="field_hat"):
        super().__init__(domain, field, name)
        self.domain_no_hat = domain
        self.domain = domain.hat()

    @classmethod
    def FromField(cls, field):
        out = cls(field.domain, field.field, field.name + "_hat")
        out.domain_no_hat = field.domain
        out.domain = field.domain.hat()
        out.field = jnp.fft.fftn(
            field.field, axes=out.all_periodic_dimensions(), norm="ortho"
        )
        return out

    def diff(self, direction, order=1):
        if direction in self.all_periodic_dimensions():
            out_field = (1j * self.domain.mgrid[direction]) ** order * self.field
        else:
            out_field = super().diff(direction, order).field
        return FourierField(
            self.domain_no_hat,
            out_field,
            name=self.name + "_diff_" + str(order),
        )

    def integrate(self, direction, order=1, bc_right=None, bc_left=None):
        if direction in self.all_periodic_dimensions():
            assert (
                type(bc_right) == NoneType and type(bc_left) == NoneType
            ), "Providing boundary conditions for integration along periodic direction makes no sense"
            mgrid = self.domain.mgrid[direction]
            field = self.field
            for i in reversed(self.all_periodic_dimensions()):
                N = mgrid.shape[i]
                inds = jnp.array(list(range(1, N)))
                mgrid = mgrid.take(indices=inds, axis=i)
                field = field.take(indices=inds, axis=i)

            denom = (1j * mgrid) ** order
            out_0 = 0.0
            out_field = jnp.pad(
                field / denom,
                [(1, 0) for _ in self.all_periodic_dimensions()],
                mode="constant",
                constant_values=out_0,
            )
        else:
            if order == 2:
                out_field = (
                    super()
                    .integrate(direction, order, bc_right=bc_right, bc_left=bc_left)
                    .field
                )
            else:
                raise NotImplementedError()
        return FourierField(
            self.domain_no_hat, out_field, name=self.name + "_int_" + str(order)
        )

    def no_hat(self):
        out = jnp.fft.ifftn(
            self.field, axes=self.all_periodic_dimensions(), norm="ortho"
        )
        return Field(self.domain_no_hat, out, name=(self.name).replace("_hat", ""))

class FourierFieldSlice(FourierField):
    def __init__(self, domain, non_periodic_direction, field, name="field", *ks):
        self.domain = domain
        self.domain_no_hat = domain
        self.non_periodic_direction = non_periodic_direction
        self.field = field
        self.ks_raw = list(ks)
        self.ks = list(ks)
        self.ks.insert(non_periodic_direction, None)
        self.name = name

    def all_dimensions(self):
        return range(len(self.ks))

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if d not in self.all_nonperiodic_dimensions()
        ]

    def all_nonperiodic_dimensions(self):
        return [
            self.non_periodic_direction
        ]

    def diff(self, direction, order=1):
        if direction in self.all_periodic_dimensions():
            out_field = (1j * self.ks[direction]) ** order * self.field
        else:
            # out_field = super().diff(1, order).field
            out_field = self.domain.diff(self.field, 0, order)
        return FourierFieldSlice(
            self.domain_no_hat,
            self.non_periodic_direction,
            out_field,
            self.name + "_diff_" + str(order),
            *self.ks_raw
        )

    def integrate(self, direction, order=1, bc_right=None, bc_left=None):
        if direction in self.all_periodic_dimensions():
            out_field = self.field / (1j * self.ks[direction]) ** order
        else:
            # out_field = super().integrate(1, order).field
            out_field = self.domain.integrate(self.field, 0, order)
        return FourierFieldSlice(
            self.domain_no_hat,
            self.non_periodic_direction,
            out_field,
            self.name + "_int_" + str(order),
            *self.ks_raw
        )

    def solve_poisson(self):
        rhs_hat = self.field
        y_mat = self.get_cheb_mat_2_homogeneous_dirichlet(0)
        n = y_mat.shape[0]
        factor = 0
        for direction in self.all_periodic_dimensions():
            factor += (1j * self.ks[direction]) ** 2

        mat = factor * jnp.eye(n) + y_mat # TODO enforce BCs before adding ks?
        mat_inv = jnp.linalg.inv(mat)

        out_field = mat_inv @ rhs_hat
        out_fourier = FourierFieldSlice(self.domain, self.non_periodic_direction, out_field, self.name + "_poisson", *self.ks_raw)
        # self.field = out_fourier.field
        return out_fourier

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        try:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        except Exception:
            new_name = "field"
        return FourierFieldSlice(self.domain_no_hat,
                                 self.non_periodic_direction, self.field + other.field, new_name,
                                 *self.ks_raw)

    def __sub__(self, other):
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            try:
                new_name = self.name + " * " + other.name
            except Exception:
                new_name = "field"
            return FourierFieldSlice(self.domain_no_hat,
                                     self.non_periodic_direction,
                                     self.field * other.field, new_name,
                                     *self.ks_raw)
        else:
            try:
                if other.real >= 0:
                    new_name = str(other) + self.name
                elif other == 1:
                    new_name = self.name
                elif other == -1:
                    new_name = "-" + self.name
                else:
                    new_name = "(" + str(other) + ") " + self.name
            except Exception:
                new_name = "field"
            return FourierFieldSlice(self.domain_no_hat,
                                     self.non_periodic_direction,
                                     self.field * other, new_name,
                                     *self.ks_raw)

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other):
        if type(other) == Field:
            raise Exception("Don't know how to divide by another field")
        else:
            try:
                if other.real >= 0:
                    new_name = self.name + "/" + other
                elif other == 1:
                    new_name = self.name
                elif other == -1:
                    new_name = "-" + self.name
                else:
                    new_name = self.name + "/ (" + str(other) + ") "
            except Exception:
                new_name = "field"
            return FourierFieldSlice(self.domain_no_hat,
                                     self.non_periodic_direction,
                                     self.field * other, new_name,
                                     *self.ks_raw)

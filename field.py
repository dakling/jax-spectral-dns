#!/usr/bin/env python3

import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    print("Unable to load")
from domain import Domain

NoneType = type(None)


class Field:
    plotting_dir = "./plots/"
    performance_mode = True

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
        for i in jnp.arange(zero_field.number_of_dofs()):
            key, subkey = jax.random.split(key)
            rands.append(
                jax.random.uniform(subkey, minval=interval[0], maxval=interval[1])
            )
        field = jnp.array(rands).reshape(zero_field.domain.shape)
        return cls(domain, field, name)

    @classmethod
    def FromField(cls, domain, field):
        fn = lambda X: field.eval(X)
        return Field.FromFunc(domain, fn, field.name + "_projected")

    def __repr__(self):
        # self.plot()
        return str(self.field)

    def __getitem__(self, index):
        return self.field[index]

    def max(self):
        return max(self.field.flatten())

    def min(self):
        return min(self.field.flatten())

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        if self.performance_mode:
            new_name = ""
        else:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        return Field(self.domain, self.field + other.field, name=new_name)

    def __sub__(self, other):
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            if self.performance_mode:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            return Field(self.domain, self.field * other.field, name=new_name)
        else:
            if self.performance_mode:
                new_name = ""
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
            if self.performance_mode:
                new_name = ""
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
        # TODO use integration or something more sophisticated
        return jnp.linalg.norm(self.field) / self.number_of_dofs()

    def l2error(self, fn):
        # fine_resolution = tuple(map(lambda x: x*10, self.domain.shape))
        # fine_domain = Domain(fine_resolution, self.domain.periodic_directions)
        # analytical_solution = Field.FromFunc(fine_domain, fn)
        # fine_field = Field.FromFunc(fine_domain, lambda X: self.eval(*X) + 0.0*X[0] * X[1] * X[2])
        # TODO supersampling
        analytical_solution = Field.FromFunc(self.domain, fn)
        return jnp.linalg.norm((self - analytical_solution).field, None)

    def eval(self, *X):
        """Evaluate field at arbitrary point X through linear interpolation. (This could obviously be improved for Chebyshev dirctions, but this is not yet implemented)"""
        grd = self.domain.grid
        print(grd)
        print(len(grd))
        print(X)
        print(len(X))
        print(len(X[0]))
        interpolant = []
        weights = []
        for dim in self.all_dimensions():
            for i in jnp.arange(len(grd[dim])):
                if (grd[dim][i] - X[dim]) * (grd[dim][i + 1] - X[dim]) <= 0:
                    interpolant.append(i)
                    weights.append(
                        (grd[dim][i] - X[dim]) / (grd[dim][i] - grd[dim][i + 1])
                    )
                    break
        base_value = self.field
        other_values = []
        for dim in self.all_dimensions():
            other_values.append(self.field)
        for dim in reversed(self.all_dimensions()):
            base_value = jnp.take(base_value, indices=interpolant[dim], axis=dim)
            for i in self.all_dimensions():
                if i == dim:
                    other_values[i] = jnp.take(
                        other_values[i], indices=interpolant[dim] + 1, axis=dim
                    )
                else:
                    other_values[i] = jnp.take(
                        other_values[i], indices=interpolant[dim], axis=dim
                    )
        out = base_value
        for dim in self.all_dimensions():
            out += (other_values[dim] - base_value) * (1 - weights[dim])
        return out

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
                    "--",
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
                    "--",
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
                    "--",
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
                    "--",
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
                        "--",
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
                        "--",
                        label=other_field.name,
                    )
            fig.legend()
            fig.savefig(self.plotting_dir + "plot_" + self.name + ".pdf")
        else:
            raise Exception("Not implemented yet")

    def plot_3d(self):
        assert (
            self.domain.number_of_dimensions == 3
        ), "Only 3D supported for this plotting method."
        # fig, ax = plt.subplots(1, 3)
        fig = plt.figure(layout="constrained")
        grd = (8, 8)
        ax = [
            plt.subplot2grid(grd, (0, 0), rowspan=2, colspan=6),
            plt.subplot2grid(grd, (2, 0), rowspan=6, colspan=6),
            plt.subplot2grid(grd, (2, 6), rowspan=6, colspan=2),
        ]
        # grd = (10, 6)
        # ax = [
        #     plt.subplot2grid(grd, (0, 0), rowspan=2, colspan=6),
        #     plt.subplot2grid(grd, (2, 0), rowspan=6, colspan=6),
        #     plt.subplot2grid(grd, (8, 0), rowspan=2, colspan=6),
        # ]
        ims = []
        for dim in self.all_dimensions():
            N_c = len(self.domain.grid[dim]) // 2
            other_dim = [i for i in self.all_dimensions() if i != dim]
            ims.append(
                ax[dim].imshow(
                    self.field.take(indices=N_c, axis=dim),
                    # labels=dict(x="xyz"[other_dim[0]], y="xyz"[other_dim[1]], color="Productivity"),
                    # x=self.domain.grid[other_dim[0]],
                    # y=self.domain.grid[other_dim[1]],
                    interpolation=None,
                    extent=[
                        self.domain.grid[other_dim[1]][0],
                        self.domain.grid[other_dim[1]][-1],
                        self.domain.grid[other_dim[0]][0],
                        self.domain.grid[other_dim[0]][-1],
                    ],
                )
            )
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in ims)
        vmax = max(image.get_array().max() for image in ims)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in ims:
            im.set_norm(norm)
        fig.colorbar(ims[0], ax=ax, label=self.name)
        fig.savefig(self.plotting_dir + "plot_3d_" + self.name + ".pdf")

    def all_dimensions(self):
        return self.domain.all_dimensions()

    def is_periodic(self, direction):
        return self.domain.is_periodic(direction)

    def all_periodic_dimensions(self):
        return self.domain.all_periodic_dimensions()

    def all_nonperiodic_dimensions(self):
        return self.domain.all_nonperiodic_dimensions()

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
                jnp.arange(len(self.domain.grid[dim]))[1:-1],
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
        name_suffix = "".join([["x", "y", "z"][direction] for _ in jnp.arange(order)])
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
                inds = jnp.arange(1, N)
                mgrid = mgrid.take(indices=inds, axis=i)

            denom += (1j * mgrid) ** 2
        out_0 = 0.0
        field = rhs_hat.field
        for i in reversed(self.all_periodic_dimensions()):
            N = field.shape[i]
            inds = jnp.arange(1, N)
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

    def max(self):
        return max([max(f) for f in self])

    def min(self):
        return min([min(f) for f in self])

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        if self.performance_mode:
            new_name = ""
        else:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        fields = []
        for i in jnp.arange(len(self)):
            fields.append(self[i] + other[i])

        out = VectorField(fields)
        out.name = new_name
        return out

    def __sub__(self, other):
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            if self.performance_mode:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            fields = []
            for i in jnp.arange(len(self)):
                fields.append(self[i] * other[i])

            out = VectorField(fields)
            out.name = new_name
            return out
        else:
            if self.performance_mode:
                new_name = ""
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
            fields = []
            for i in jnp.arange(len(self)):
                fields.append(self[i] * other)

            out = VectorField(fields)
            out.name = new_name
            return out

    def __truediv__(self, other):
        if type(other) == Field:
            raise Exception("Don't know how to divide by another field")
        else:
            if self.performance_mode:
                new_name = ""
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
            fields = []
            for i in jnp.arange(len(self)):
                fields.append(self[i] / other)

            out = VectorField(fields)
            out.name = new_name
            return out

    def __abs__(self):
        out = 0
        for f in self:
            out += abs(f)
        return out

    def all_dimensions(self):
        return self[0].domain.all_dimensions()
        # return jnp.arange(self[0].domain.number_of_dimensions)

    def all_periodic_dimensions(self):
        return self[0].domain.all_periodic_dimensions()
        # return [
        #     self.all_dimensions()[d]
        #     for d in self.all_dimensions()
        #     if self[0].domain.periodic_directions[d]
        # ]

    def all_nonperiodic_dimensions(self):
        return self[0].domain.all_nonperiodic_dimensions()
        # return [
        #     self.all_dimensions()[d]
        #     for d in self.all_dimensions()
        #     if not self[0].domain.periodic_directions[d]
        # ]

    def hat(self):
        return VectorField([f.hat() for f in self])

    def no_hat(self):
        return VectorField([f.no_hat() for f in self])

    def plot(self, *other_fields):
        for i in jnp.arange(len(self)):
            other_fields_i = [item[i] for item in other_fields]
            self[i].plot(*other_fields_i)

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

    def reconstruct_from_wavenumbers(self, fn):
        assert self.number_of_dimensions != 3, "2D not implemented yet"
        k1 = self[0].domain.grid[self.all_periodic_dimensions()[0]].astype(int)
        k2 = self[0].domain.grid[self.all_periodic_dimensions()[1]].astype(int)
        k1_ints = jnp.arange(len(k1))
        k2_ints = jnp.arange(len(k2))
        jit_fn = jax.jit(fn)
        # jit_fn = (fn)
        out_array = [[jit_fn(k1_, k2_) for k2_ in k2] for k1_ in k1]
        # out_array = jnp.array(jax.vmap(lambda k2_: jax.vmap(lambda k1_: jit_fn(k1_, k2_))(k1))(k2))
        out_field = [
            FourierField(
                self[0].domain_no_hat,
                jnp.moveaxis(
                    # jnp.array([[out_array[k1_][k2_][i] for k2_ in k2] for k1_ in k1]),
                    jnp.array([[out_array[k1_][k2_][i] for k2_ in k2_ints] for k1_ in k1_ints]),
                    # jnp.array(jax.vmap(lambda k1_: jax.vmap(lambda k2_: out_array.at[k1_,k2_,i].get())(k2))(k1)),
                    # jnp.array(jax.vmap(lambda k1_: jax.vmap(lambda k2_: out_array[k1_][k2_].get()[i])(k2))(k1)),
                    -1,
                    self.all_nonperiodic_dimensions()[0],
                ),
                "out_" + str(i),
            )
            for i in self.all_dimensions()
        ]
        return VectorField(out_field)


class FourierField(Field):
    def __init__(self, domain, field, name="field_hat"):
        super().__init__(domain, field, name)
        self.domain_no_hat = domain
        self.domain = domain.hat()

    def __add__(self, other):
        out = super().__add__(other)
        return FourierField(self.domain_no_hat, out.field, name=out.name)

    def __mul__(self, other):
        out = super().__mul__(other)
        return FourierField(self.domain_no_hat, out.field, name=out.name)

    def __truediv__(self, other):
        out = super().__truediv__(other)
        return FourierField(self.domain_no_hat, out.field, name=out.name)

    @classmethod
    def FromField(cls, field):
        out = cls(field.domain, field.field, field.name + "_hat")
        out.domain_no_hat = field.domain
        out.domain = field.domain.hat()

        scaling_factor = 1.0
        for i in out.all_periodic_dimensions():
            scaling_factor *= out.domain.scale_factors[i] / (2 * jnp.pi)

        out.field = jnp.fft.fftn(
            field.field, axes=list(out.all_periodic_dimensions()), norm="ortho"
        ) / scaling_factor
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
                inds = jnp.arange(1, N)
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

        scaling_factor = 1.0
        for i in self.all_periodic_dimensions():
            scaling_factor *= self.domain.scale_factors[i] / (2 * jnp.pi)

        out = jnp.fft.ifftn(
            self.field, axes=self.all_periodic_dimensions(), norm="ortho"
        ).real / (1 / scaling_factor)
        return Field(self.domain_no_hat, out, name=(self.name).replace("_hat", ""))

    def reconstruct_from_wavenumbers(self, fn):
        assert self.number_of_dimensions != 3, "2D not implemented yet"
        k1 = self.domain.grid[self.all_periodic_dimensions()[0]]
        k2 = self.domain.grid[self.all_periodic_dimensions()[1]]
        out_field = jnp.moveaxis(
            jnp.array([[fn(k1_, k2_) for k2_ in k2] for k1_ in k1]),
            -1,
            self.all_nonperiodic_dimensions()[0],
        )
        return FourierField(self.domain_no_hat, out_field, name=self.name + "_reconstr")


class FourierFieldSlice(FourierField):
    def __init__(self, domain, non_periodic_direction, field, name="field", *ks):
        self.domain = domain
        self.domain_no_hat = domain
        self.non_periodic_direction = non_periodic_direction
        self.field = field
        self.ks_raw = list(ks)
        self.ks = jnp.array(ks)
        self.ks = jnp.insert(self.ks, non_periodic_direction, -1)
        self.name = name

    def all_dimensions(self):
        return range(len(self.ks))
        # return jnp.arange(len(self.ks))

    def all_periodic_dimensions(self):
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if d not in self.all_nonperiodic_dimensions()
        ]

    def all_nonperiodic_dimensions(self):
        return [self.non_periodic_direction]

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

        # unit matrix with zeros in first and  last row/col to avoid messing with bcs
        eye_bc = jnp.block(
            [
                [jnp.zeros((1, n))],
                [jnp.zeros((n - 2, 1)), jnp.eye(n - 2), jnp.zeros((n - 2, 1))],
                [jnp.zeros((1, n))],
            ]
        )
        mat = factor * eye_bc + y_mat
        mat_inv = jnp.linalg.inv(mat)
        out_field = mat_inv @ rhs_hat
        out_fourier = FourierFieldSlice(
            self.domain,
            self.non_periodic_direction,
            out_field,
            self.name + "_poisson",
            *self.ks_raw
        )
        # self.field = out_fourier.field
        return out_fourier

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        if self.performance_mode:
            new_name = ""
        else:
            try:
                if other.name[0] == "-":
                    new_name = self.name + " - " + other.name[1:]
                else:
                    new_name = self.name + " + " + other.name
            except Exception:
                new_name = "field"
        return FourierFieldSlice(
            self.domain_no_hat,
            self.non_periodic_direction,
            self.field + other.field,
            new_name,
            *self.ks_raw
        )

    def __sub__(self, other):
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            if self.performance_mode:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            return FourierFieldSlice(
                self.domain_no_hat,
                self.non_periodic_direction,
                self.field * other.field,
                new_name,
                *self.ks_raw
            )
        else:
            if self.performance_mode:
                new_name = ""
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
            return FourierFieldSlice(
                self.domain_no_hat,
                self.non_periodic_direction,
                self.field * other,
                new_name,
                *self.ks_raw
            )

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other):
        if type(other) == Field:
            raise Exception("Don't know how to divide by another field")
        else:
            if self.performance_mode:
                new_name = ""
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
            return FourierFieldSlice(
                self.domain_no_hat,
                self.non_periodic_direction,
                self.field / other,
                new_name,
                *self.ks_raw
            )

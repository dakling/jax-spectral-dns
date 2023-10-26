#!/usr/bin/env python3

import time

import math
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.figure as figure
from matplotlib import colors
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator

import numpy as np

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    print("Unable to load")
from domain import Domain

NoneType = type(None)


@partial(jax.jit, static_argnums=(0, 1))
def reconstruct_from_wavenumbers_jit(domain, fn):
    time_1 = time.time()
    # vectorize = False
    vectorize = True
    k1s = jnp.array(domain.grid[domain.all_periodic_dimensions()[0]].astype(int))
    k2s = jnp.array(domain.grid[domain.all_periodic_dimensions()[1]].astype(int))
    k1_ints = jnp.arange(len(k1s))
    k2_ints = jnp.arange(len(k2s))
    jit_fn = jax.jit(fn)
    # jit_fn = fn

    # previously best varant using list comprehensions
    if vectorize == False:
        Nx, Ny, Nz = len(k1_ints), -1, len(k2_ints)
        # t1 = time.time()
        out_array = jnp.array(
            [[jit_fn((k1_, k2_)) for k2_ in k2_ints] for k1_ in k1_ints]
        )
        out_field = jnp.array(
            [
                jnp.moveaxis(
                    out_array[:, :, i, :],
                    -1,
                    domain.all_nonperiodic_dimensions()[0],
                )
                for i in domain.all_dimensions()
            ]
        )

    # might be better on GPUs? TODO: fix bug
    else:
        time_2 = time.time()
        K1, K2 = jnp.meshgrid(k1_ints, k2_ints)
        # K = jnp.array(list(zip(K1.flatten(), K2.flatten())))
        K = jax.lax.map(lambda ks: (ks[0], ks[1]), [K1.flatten(), K2.flatten()])
        jit_fn_vec = jax.vmap(jit_fn)
        # jit_fn_vec = jax.vmap(fn)
        Nx, Ny, Nz = len(k1_ints), -1, len(k2_ints)
        out_array = jnp.array(jit_fn_vec(K))
        out_field = jax.lax.map(
            lambda i: jnp.moveaxis(
                out_array[i, :, :].reshape(Nx, Nz, Ny),
                -1,
                domain.all_nonperiodic_dimensions()[0],
            ),
            jnp.arange(domain.number_of_dimensions),
        )
    return out_field


class Field:
    """Class that holds the information needed to describe a dependent variable
    and to perform operations on it."""
    plotting_dir = "./plots/"
    # plotting_format = ".pdf"
    plotting_format = ".png"
    field_dir = "./fields/"
    performance_mode = True

    def __init__(self, domain, field, name="field"):
        self.domain = domain
        self.domain_no_hat = domain
        self.field = field
        self.name = name
        self.time_step = 0

    @classmethod
    def FromFunc(cls, domain, func=None, name="field"):
        """Construct from function func depending on the independent variables described by domain."""
        if not func:
            func = lambda x: 0.0 * math.prod(x)
        field = jnp.array(list(map(lambda *x: func(x), *domain.mgrid)))
        return cls(domain, field, name)

    @classmethod
    def FromRandom(cls, domain, seed=0, interval=(-0.1, 0.1), name="field"):
        """Construct a random field depending on the independent variables described by domain."""
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
        """Construct a new field depending on the independent variables described by
        domain by interpolating a given field."""
        # TODO testing + performance improvements needed
        assert not isinstance(field, FourierField), "Attempted to interpolate a Field from a FourierField."
        out = []
        if domain.number_of_dimensions == 1:
            for x in domain.grid[0]:
                out.append(field.eval(x))
        elif domain.number_of_dimensions == 2:
            for x in domain.grid[0]:
                for y in domain.grid[1]:
                    out.append(field.eval([x, y]))
            out = jnp.array(out)
            out.reshape(domain.number_of_cells(0), domain.number_of_cells(1))
        elif domain.number_of_dimensions == 3:
            for x in domain.grid[0]:
                for y in domain.grid[1]:
                    for z in domain.grid[2]:
                        out.append(field.eval([x, y, z]))
            out = jnp.array(out)
            out.reshape(domain.number_of_cells(0), domain.number_of_cells(1), domain.number_of_cells(2))
        else:
            raise NotImplementedError("Number of dimensions not supported.")
        return Field(domain, out, field.name + "_projected")

    @classmethod
    def FromFile(cls, domain, filename, name="field"):
        """Construct new field depending on the independent variables described
        by domain by reading in a saved field from file filename."""
        out = Field(domain, None, name=name)
        field_array = np.load(out.field_dir + filename, allow_pickle=True)
        out.field = jnp.array(field_array.tolist())
        return out

    def save_to_file(self, filename):
        """Save field to file filename."""
        field_array = np.array(self.field.tolist())
        field_array.dump(self.field_dir + filename)

    def get_name(self):
        """Return the name of the field."""
        return self.name

    def set_name(self, name):
        """Set the name of the field."""
        self.name = name

    def get_time_step(self):
        """Return the current time step of the field."""
        return self.time_step

    def set_time_step(self, time_step):
        """Set the current time step of the field."""
        self.time_step = time_step

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
        ret = self * (-1.0)
        ret.time_step = self.time_step
        return ret

    def __add__(self, other):
        assert not isinstance(other, FourierField), "Attempted to add a Field and a Fourier Field."
        if self.performance_mode:
            new_name = ""
        else:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        ret = Field(self.domain, self.field + other.field, name=new_name)
        ret.time_step = self.time_step
        return ret

    def __sub__(self, other):
        assert not isinstance(other, FourierField), "Attempted to subtract a Field and a Fourier Field."
        return self + other * (-1.0)

    def __mul__(self, other):
        if isinstance(other, Field):
            assert not isinstance(other, FourierField), "Attempted to multiply a Field and a Fourier Field."
            if self.performance_mode:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            ret = Field(self.domain, self.field * other.field, name=new_name)
            ret.time_step = self.time_step
            return ret
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
            ret = Field(self.domain, self.field * other, name=new_name)
            ret.time_step = self.time_step
            return ret

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
            ret = Field(self.domain, self.field * other, name=new_name)
            ret.time_step = self.time_step
            return ret

    def shift(self, value):
        out_field = self.field + value
        return Field(self.domain, out_field, name=self.name)

    def __abs__(self):
        # TODO use integration or something more sophisticated
        return jnp.linalg.norm(self.field) / self.number_of_dofs()

    def l2error(self, fn):
        # TODO supersampling
        analytical_solution = Field.FromFunc(self.domain, fn)
        return jnp.linalg.norm((self - analytical_solution).field, None)

    def volume_integral(self):
        int = Field(self.domain, self.field)
        for i in reversed(self.all_dimensions()):
            int = int.definite_integral(i)
        return int

    def energy(self):
        energy = 0.5 * self * self
        return energy.volume_integral()

    def eval(self, X):
        """Evaluate field at arbitrary point X through linear interpolation. (TODO: This could obviously be improved for Chebyshev dirctions, but this is not yet implemented)"""
        grd = self.domain.grid
        interpolant = []
        weights = []
        for dim in self.all_dimensions():
            for i in jnp.arange(self.domain.number_of_cells(dim)):
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
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            ax.plot(self.domain.grid[0], self.field, label=self.name)
            for other_field in other_fields:
                ax.plot(
                    self.domain.grid[dimension],
                    other_field.field,
                    "--",
                    label=other_field.name,
                )
            fig.legend()
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_latest"
                + self.plotting_format
            )
            # fig.savefig(self.plotting_dir + "plot_cl_" + self.name + "_t_" + str(self.time_step) + self.plotting_format)
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )
            # plt.close(fig)
        elif self.domain.number_of_dimensions == 2:
            # fig, ax = plt.subplots(1, 1)
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            other_dim = [i for i in self.all_dimensions() if i != dimension][0]
            N_c = self.domain.number_of_cells(other_dim) // 2
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
                + "_latest"
                + self.plotting_format
            )
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_"
                + ["x", "y"][dimension]
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )
            # plt.close(fig)
        elif self.domain.number_of_dimensions == 3:
            # fig, ax = plt.subplots(1, 1)
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            other_dim = [i for i in self.all_dimensions() if i != dimension]
            N_c = [self.domain.number_of_cells(dim) // 2 for dim in other_dim]
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
                + "_latest"
                + self.plotting_format
            )
            fig.savefig(
                self.plotting_dir
                + "plot_cl_"
                + self.name
                + "_"
                + ["x", "y", "z"][dimension]
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )
        else:
            raise Exception("Not implemented yet")

    def plot(self, *other_fields):
        if self.domain.number_of_dimensions == 1:
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
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
            fig.savefig(
                self.plotting_dir
                + "plot_"
                + self.name
                + "_latest"
                + self.plotting_format
            )
            fig.savefig(
                self.plotting_dir
                + "plot_"
                + self.name
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )

        elif self.domain.number_of_dimensions == 2:
            fig = figure.Figure(figsize=(15, 5))
            ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)]
            ax3d = fig.add_subplot(1, 3, 3, projection="3d")
            for dimension in self.all_dimensions():
                other_dim = [i for i in self.all_dimensions() if i != dimension][0]
                N_c = self.domain.number_of_cells(other_dim) // 2
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
                fig.savefig(
                    self.plotting_dir
                    + "plot_"
                    + self.name
                    + "_latest"
                    + self.plotting_format
                )
                fig.savefig(
                    self.plotting_dir
                    + "plot_"
                    + self.name
                    + "_t_"
                    + "{:06}".format(self.time_step)
                    + self.plotting_format
                )
        elif self.domain.number_of_dimensions == 3:
            fig = figure.Figure()
            # ax = fig.subplots(1, 3, figsize=(15, 5))
            ax = fig.subplots(1, 3)
            for dimension in self.all_dimensions():
                other_dim = [i for i in self.all_dimensions() if i != dimension]
                N_c = [self.domain.number_of_cells(dim) // 2 for dim in other_dim]
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
            fig.savefig(
                self.plotting_dir
                + "plot_"
                + self.name
                + "_latest"
                + self.plotting_format
            )
            fig.savefig(
                self.plotting_dir
                + "plot_"
                + self.name
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )
        else:
            raise Exception("Not implemented yet")

    def plot_3d(self, direction=None):
        if type(direction) != NoneType:
            self.plot_3d_single(direction)
        else:
            assert (
                self.domain.number_of_dimensions == 3
            ), "Only 3D supported for this plotting method."
            fig = figure.Figure(layout="constrained")
            base_len = 100
            grd = (base_len, base_len)
            lx = self.domain.scale_factors[0]
            ly = self.domain.scale_factors[1] * 2
            lz = self.domain.scale_factors[2]
            rows_x = int(ly / (ly + lx) * base_len)
            cols_x = int(lz / (lz + ly) * base_len)
            rows_y = int(lx / (ly + lx) * base_len)
            cols_y = int(lz / (lz + ly) * base_len)
            rows_z = int(lx / (ly + lx) * base_len)
            cols_z = int(ly / (lz + ly) * base_len)
            ax = [
                fig.add_subplot(fig.add_gridspec(*grd)[0 : 0 + rows_x, 0 : 0 + cols_x]),
                fig.add_subplot(
                    fig.add_gridspec(*grd)[rows_x : rows_x + rows_y, 0 : 0 + cols_y]
                ),
                fig.add_subplot(
                    fig.add_gridspec(*grd)[
                        rows_x : rows_x + rows_z, cols_y : cols_y + cols_z
                    ]
                ),
            ]
            ims = []
            for dim in self.all_dimensions():
                N_c = self.domain.number_of_cells(dim) // 2
                other_dim = [i for i in self.all_dimensions() if i != dim]
                ims.append(
                    ax[dim].imshow(
                        self.field.take(indices=N_c, axis=dim),
                        interpolation=None,
                        extent=[
                            self.domain.grid[other_dim[1]][0],
                            self.domain.grid[other_dim[1]][-1],
                            self.domain.grid[other_dim[0]][0],
                            self.domain.grid[other_dim[0]][-1],
                        ],
                    )
                )
                ax[dim].set_xlabel("xyz"[other_dim[1]])
                ax[dim].set_ylabel("xyz"[other_dim[0]])
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in ims)
            vmax = max(image.get_array().max() for image in ims)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in ims:
                im.set_norm(norm)
            fig.colorbar(ims[0], ax=ax, label=self.name)
            fig.savefig(
                self.plotting_dir
                + "plot_3d_"
                + self.name
                + "_latest"
                + self.plotting_format
            )
            fig.savefig(
                self.plotting_dir
                + "plot_3d_"
                + self.name
                + "_t_"
                + "{:06}".format(self.time_step)
                + self.plotting_format
            )

    def plot_3d_single(self, dim):
        assert (
            self.domain.number_of_dimensions == 3
        ), "Only 3D supported for this plotting method."
        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        ims = []
        N_c = self.domain.number_of_cells(dim) // 2
        other_dim = [i for i in self.all_dimensions() if i != dim]
        ims.append(
            ax.imshow(
                self.field.take(indices=N_c, axis=dim).T,
                interpolation=None,
                extent=[
                    self.domain.grid[other_dim[0]][0],
                    self.domain.grid[other_dim[0]][-1],
                    self.domain.grid[other_dim[1]][0],
                    self.domain.grid[other_dim[1]][-1],
                ],
            )
        )
        ax.set_xlabel("xyz"[other_dim[0]])
        ax.set_ylabel("xyz"[other_dim[1]])
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in ims)
        vmax = max(image.get_array().max() for image in ims)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in ims:
            im.set_norm(norm)
        fig.colorbar(ims[0], ax=ax, label=self.name, orientation="vertical")
        fig.savefig(
            self.plotting_dir
            + "plot_3d_"
            + "xyz"[dim]
            + "_"
            + self.name
            + "_latest"
            + self.plotting_format
        )
        fig.savefig(
            self.plotting_dir
            + "plot_3d_"
            + "xyz"[dim]
            + "_"
            + self.name
            + "_t_"
            + "{:06}".format(self.time_step)
            + self.plotting_format
        )

    def plot_isolines(self, normal_direction, isolines=None):
        if type(isolines) == NoneType:
            isolines = [0, 1.5, 2.5, 3.5]
            isolines += [- i for i in isolines[1:]]

        isolines.sort()

        fig = figure.Figure()
        ax = fig.subplots(1,1)
        directions = [i for i in self.all_dimensions() if i != normal_direction]
        x = self.domain.grid[directions[0]]
        y = self.domain.grid[directions[1]]
        X, Y = jnp.meshgrid(x, y)
        N_c = self.domain.number_of_cells(normal_direction) // 2
        f = self.field.take(indices=N_c, axis=normal_direction).T
        cmap = colors.ListedColormap([('gray', 0.3), 'white'])
        bounds=[-1e10,0,1e10]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(f, interpolation='gaussian', origin='lower',
                  cmap=cmap, norm=norm, extent=(x[0], x[-1], y[0], y[-1]))
        CS= ax.contour(X, Y, f, isolines)
        ax.clabel(CS, inline=True, fontsize=10)
        fig.savefig(
            self.plotting_dir
            + "plot_iso_"
            + "xyz"[normal_direction]
            + "_"
            + self.name
            + "_t_"
            + "{:06}".format(self.time_step)
            + self.plotting_format
        )
        fig.savefig(
            self.plotting_dir
            + "plot_iso_"
            + "xyz"[normal_direction]
            + "_"
            + self.name
            + "_latest"
            + self.plotting_format
        )

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
                jnp.arange(self.domain.number_of_cells(dim))[1:-1],
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
        out = FourierField.FromField(self)
        out.time_step = self.time_step
        return out

    def diff(self, direction, order=1):
        name_suffix = "".join([["x", "y", "z"][direction] for _ in jnp.arange(order)])
        return Field(
            self.domain,
            self.domain.diff(self.field, direction, order),
            self.name + "_" + name_suffix,
        )

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction):
        return self.domain.get_cheb_mat_2_homogeneous_dirichlet(direction)

    def get_cheb_mat_2_homogeneous_dirichlet_only_rows(self, direction):
        return self.domain.get_cheb_mat_2_homogeneous_dirichlet_only_rows(direction)

    def integrate(self, direction, order=1, bc_left=None, bc_right=None):
        out_bc = self.domain.integrate(self.field, direction, order, bc_left, bc_right)
        return Field(self.domain, out_bc, name=self.name + "_int")

    def definite_integral(self, direction):
        if not self.is_periodic(direction):
            int = self.integrate(direction, 1, bc_right=0.0)
            if self.number_of_dimensions() == 1:
                return int[0] - int[-1]
            else:
                N = self.domain.number_of_cells(direction)
                inds = [i for i in self.all_dimensions() if i != direction]
                shape = tuple((jnp.array(self.domain.shape)[tuple(inds),]).tolist())
                periodic_directions = tuple(
                    (jnp.array(self.domain.periodic_directions)[tuple(inds),]).tolist()
                )
                scale_factors = tuple(
                    (jnp.array(self.domain.scale_factors)[tuple(inds),]).tolist()
                )
                reduced_domain = Domain(
                    shape, periodic_directions, scale_factors=scale_factors
                )
                field = jnp.take(int.field, indices=0, axis=direction) - jnp.take(
                    int.field, indices=N - 1, axis=direction
                )
                return Field(reduced_domain, field)
        else:
            N = self.domain.number_of_cells(direction)
            if self.number_of_dimensions() == 1:
                return self.domain.scale_factors[direction] / N * jnp.sum(self.field[:])
            else:
                N = self.domain.number_of_cells(direction)
                inds = [i for i in self.all_dimensions() if i != direction]
                shape = tuple((jnp.array(self.domain.shape)[tuple(inds),]).tolist())
                periodic_directions = tuple(
                    (jnp.array(self.domain.periodic_directions)[tuple(inds),]).tolist()
                )
                scale_factors = tuple(
                    (jnp.array(self.domain.scale_factors)[tuple(inds),]).tolist()
                )
                reduced_domain = Domain(
                    shape, periodic_directions, scale_factors=scale_factors
                )
                field = (
                    self.domain.scale_factors[direction]
                    / N
                    * np.add.reduce(self.field, axis=direction)
                )
                return Field(reduced_domain, field)

    def nabla(self):
        out = [self.diff(0)]
        out[0].name = "nabla_" + self.name + "_" + str(0)
        for dim in self.all_dimensions()[1:]:
            out.append(self.diff(dim))
            out[dim].name = "nabla_" + self.name + "_" + str(dim)
            out[dim].time_step = self.time_step
        return VectorField(out)

    def laplacian(self):
        out = self.diff(0, 2)
        for dim in self.all_dimensions()[1:]:
            out += self.diff(dim, 2)
        out.name = "lap_" + self.name
        out.time_step = self.time_step
        return out

    def perform_explicit_euler_step(self, eq, dt, i):
        new_u = self + eq * dt
        new_u.update_boundary_conditions()
        new_u.name = "u_" + str(i)
        return new_u

    def perform_time_step(self, eq, dt, i):
        return self.perform_explicit_euler_step(eq, dt, i)


class VectorField:
    def __init__(self, elements, name=None):
        self.elements = elements
        self.name = name
        self.domain = elements[0].domain

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

    def __add__(self, other):
        out = []
        for i in range(len(self)):
            out.append(self[i] + other[i])
        return VectorField(out)

    def __sub__(self, other):
        return self + (-1) * other

    def __mul__(self, other):
        out = []
        if isinstance(other, Field):
            for i in range(len(self)):
                out.append(self[i] * other[i])
        else:
            for i in range(len(self)):
                out.append(self[i] * other)
        return VectorField(out)

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other):
        out = []
        for i in range(len(self)):
            out.append(self[i] / other)
        return VectorField(out)

    def shift(self, value):
        out = []
        assert len(value) == len(self), "Dimension mismatch."
        for i in range(len(self)):
            out.append(self[i].shift(value[i]))
        return VectorField(out)

    def max(self):
        return max([max(f) for f in self])

    def min(self):
        return min([min(f) for f in self])

    def __abs__(self):
        out = 0
        for f in self:
            out += abs(f)
        return out

    def energy(self):
        en = 0
        for f in self:
            en += f.energy()
        return en

    def get_time_step(self):
        time_steps = [f.time_step for f in self]
        return max(time_steps)

    def set_time_step(self, time_step):
        self.time_step = time_step
        for j in range(len(self)):
            self[j].time_step = time_step

    def get_name(self):
        if type(self.name) != NoneType:
            return self.name
        if len(self) == 1:
            return self[0].name
        else:
            name = self[0].name
            for i in range(len(self)):
                name = set(name).intersection(self[i].name)
            return name

    def set_name(self, name):
        self.name = name
        for j in range(len(self)):
            self[j].name = name + "_" + "xyz"[j]

    def update_boundary_conditions(self):
        for field in self:
            field.update_boundary_conditions()

    def all_dimensions(self):
        return self[0].domain.all_dimensions()

    def all_periodic_dimensions(self):
        return self[0].domain.all_periodic_dimensions()

    def all_nonperiodic_dimensions(self):
        return self[0].domain.all_nonperiodic_dimensions()

    def hat(self):
        return VectorField([f.hat() for f in self])

    def no_hat(self):
        return VectorField([f.no_hat() for f in self])

    def plot(self, *other_fields):
        for i in jnp.arange(len(self)):
            other_fields_i = [item[i] for item in other_fields]
            self[i].plot(*other_fields_i)

    def cross_product(self, other):
        out_0 = self[1] * other[2] - self[2] * other[1]
        out_1 = self[2] * other[0] - self[0] * other[2]
        out_2 = self[0] * other[1] - self[1] * other[0]

        time_step = self.get_time_step()
        out = [out_0, out_1, out_2]
        for f in out:
            f.time_step = time_step

        return VectorField(out)

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

        time_step = self.get_time_step()
        out = [curl_0, curl_1, curl_2]
        for f in out:
            f.time_step = time_step

        return VectorField(out)

    def div(self):
        out = self[0].diff(0)
        out.name = "div_" + self.name + "_" + str(0)
        for dim in self.all_dimensions()[1:]:
            out += self[dim].diff(dim)
        return out

    def reconstruct_from_wavenumbers(self, fn, number_of_other_fields=0):
        assert self.number_of_dimensions != 3, "2D not implemented yet"

        # jit = True
        # vectorize = True

        jit = False
        vectorize = False

        if jit:
            time_1 = time.time()
            out_field = reconstruct_from_wavenumbers_jit(self[0].domain_no_hat, fn)
            time_2 = time.time()

            out = VectorField(
                [
                    FourierField(self[0].domain_no_hat, out_field[i])
                    for i in self.all_dimensions()
                ]
            )
            time_3 = time.time()
            print("in reconstr: time for part 1: ", time_2 - time_1)
            print("in reconstr: time for part 2: ", time_3 - time_2)
            # out = VectorField(jax.lax.map(lambda i: FourierField(self[0].domain_no_hat, out_field.at[i].get()), jnp.arange(self[0].domain.number_of_dimensions)))
            return out
        else:
            time_1 = time.time()
            k1s = jnp.array(
                self[0].domain.grid[self.all_periodic_dimensions()[0]].astype(int)
            )
            k2s = jnp.array(
                self[0].domain.grid[self.all_periodic_dimensions()[1]].astype(int)
            )
            k1_ints = jnp.arange(len(k1s))
            k2_ints = jnp.arange(len(k2s))
            jit_fn = jax.jit(fn)
            # jit_fn = fn

            # previously best varant using list comprehensions
            if vectorize == False:
                Nx, Ny, Nz = len(k1_ints), -1, len(k2_ints)
                # t1 = time.time()
                out_array = jnp.array(
                    [[jit_fn((k1_, k2_)) for k2_ in k2_ints] for k1_ in k1_ints]
                )
                out_field = [
                    FourierField(
                        self[0].domain_no_hat,
                        jnp.moveaxis(
                            out_array[:, :, i, :],
                            -1,
                            self.all_nonperiodic_dimensions()[0],
                        ),
                        "out_" + str(i),
                    )
                    for i in self.all_dimensions()
                ]

                other_field = []
                for i in range(number_of_other_fields):
                    other_field.append(
                        FourierField(
                            self[0].domain_no_hat,
                            jnp.moveaxis(
                                out_array[:, :, i + 3, :],
                                -1,
                                self.all_nonperiodic_dimensions()[0],
                            ),
                            "out_" + str(i),
                        )
                    )

            # might be better on GPUs? TODO: fix bug
            else:
                K1, K2 = jnp.meshgrid(k1_ints, k2_ints)
                K = jnp.array(list(zip(K1.flatten(), K2.flatten())))
                jit_fn_vec = jax.vmap(jit_fn)
                Nx, Ny, Nz = len(k1_ints), -1, len(k2_ints)
                out_array = jnp.array(jit_fn_vec(K))
                out_field = [
                    FourierField(
                        self[0].domain_no_hat,
                        jnp.moveaxis(
                            out_array[i, :, :].reshape(Nx, Nz, Ny),
                            -1,
                            self.all_nonperiodic_dimensions()[0],
                        ),
                        "other_" + str(i),
                    )
                    for i in self.all_dimensions()
                ]
            return (VectorField(out_field), other_field)

    def plot_streamlines(self, normal_direction, isolines=None):
        fig = figure.Figure()
        ax = fig.subplots(1,1)
        directions = [i for i in self.all_dimensions() if i != normal_direction]
        x = self.domain.grid[directions[0]]
        y = jnp.flip(self.domain.grid[directions[1]])
        N = 40
        xi = np.linspace(x[0], x[-1], N)
        yi = np.linspace(y[0], y[-1], N)
        N_c = self.domain.number_of_cells(normal_direction) // 2
        U = self[directions[0]].field.take(indices=N_c, axis=normal_direction)
        V = self[directions[1]].field.take(indices=N_c, axis=normal_direction)
        interp_u = RegularGridInterpolator((x, y), U, method='cubic')
        interp_v = RegularGridInterpolator((x, y), V, method='cubic')
        Ui = np.array([[ interp_u([[x_, y_]])[0] for x_ in xi ] for y_ in yi])
        Vi = np.array([[ interp_v([[x_, y_]])[0] for x_ in xi ] for y_ in yi])

        ax.streamplot(xi, yi, Ui, Vi, broken_streamlines=False, linewidth=0.2)
        fig.savefig(
            self[0].plotting_dir
            + "plot_streamlines_"
            + self.get_name()
            + "_t_"
            + "{:06}".format(self.time_step)
            + self[0].plotting_format
        )
        fig.savefig(
            self[0].plotting_dir
            + "plot_streamlines_"
            + self.get_name()
            + "_latest"
            + self[0].plotting_format
        )

class FourierField(Field):
    def __init__(self, domain, field, name="field_hat"):
        super().__init__(domain, field, name)
        self.domain_no_hat = domain
        self.domain = domain.hat()

    def __add__(self, other):
        assert isinstance(other, FourierField), "Attempted to add a Fourier Field and a Field."
        if self.performance_mode:
            new_name = ""
        else:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        ret = FourierField(self.domain_no_hat, self.field + other.field, name=new_name)
        ret.time_step = self.time_step
        return ret

    def __mul__(self, other):
        if isinstance(other, Field):
            assert isinstance(other, FourierField), "Attempted to multiply a Fourier Field and a Field."

        if isinstance(other, FourierField):
            if self.performance_mode:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            ret = FourierField(self.domain_no_hat, self.field * other.field, name=new_name)
            ret.time_step = self.time_step
            return ret
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
            ret = FourierField(self.domain_no_hat, self.field * other, name=new_name)
            ret.time_step = self.time_step
            return ret

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other):
        out = super().__truediv__(other)
        return FourierField(self.domain_no_hat, out.field, name=out.name)

    @classmethod
    def FromField(cls, field):
        out = cls(field.domain, field.field, field.name + "_hat")
        out.domain_no_hat = field.domain
        out.domain = field.domain.hat()
        out.field = out.domain_no_hat.field_hat(field.field)
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
            mgrid = self.domain.mgrid[direction]
            field = self.field
            N = mgrid.shape[direction]
            inds = jnp.arange(1, N)
            mgrid = mgrid.take(indices=inds, axis=direction)
            field = field.take(indices=inds, axis=direction)

            denom = (1j * mgrid) ** order
            out_0 = 0.0
            out_field = jnp.pad(
                field / denom,
                [(1, 0) if i == direction else (0, 0) for i in self.all_dimensions()],
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
                out_field = (
                    super()
                    .integrate(direction, order, bc_right=bc_right, bc_left=bc_left)
                    .field
                )

        return FourierField(
            self.domain_no_hat, out_field, name=self.name + "_int_" + str(order)
        )

    # def laplacian(self):
    #     out = self.diff(0, 2)
    #     for dim in self.all_dimensions()[1:]:
    #         out += self.diff(dim, 2)
    #     out.name = "lap_" + self.name
    #     out.time_step = self.time_step
    #     return out


    def definite_integral(self, direction):
        raise NotImplementedError()

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
        k1 = self.domain.grid[self.all_periodic_dimensions()[0]]
        k2 = self.domain.grid[self.all_periodic_dimensions()[1]]
        k1sq = k1**2
        k2sq = k2**2
        mat = jnp.array(
            [
                [jnp.linalg.inv((-(k1sq_ + k2sq_)) * eye_bc + y_mat) for k2sq_ in k2sq]
                for k1sq_ in k1sq
            ]
        )
        return mat

    def solve_poisson(self, mat=None):
        assert len(self.all_dimensions()) == 3, "Only 3d implemented currently."
        assert (
            len(self.all_nonperiodic_dimensions()) <= 1
        ), "Poisson solution not implemented for the general case."
        rhs_hat = self.field
        if type(mat) == NoneType:
            mat = self.assemble_poisson_matrix()
        field = rhs_hat
        out_field = jnp.pad(
            jnp.einsum("ijkl,ilj->ikj", mat, field),
            [
                (0, 0) if i in self.all_periodic_dimensions() else (0, 0)
                for i in self.all_dimensions()
            ],
            mode="constant",
            constant_values=0.0,
        )
        out_fourier = FourierField(
            self.domain_no_hat, out_field, name=self.name + "_poisson"
        )
        return out_fourier

    def no_hat(self):
        out = self.domain_no_hat.no_hat(self.field)
        out_field = Field(self.domain_no_hat, out, name=(self.name).replace("_hat", ""))
        out_field.time_step = self.time_step
        return out_field

    def reconstruct_from_wavenumbers(self, fn, vectorize=False):
        if vectorize:
            print("vectorisation not implemented yet, using unvectorized version")
        assert self.number_of_dimensions != 3, "2D not implemented yet"
        k1 = self.domain.grid[self.all_periodic_dimensions()[0]]
        k2 = self.domain.grid[self.all_periodic_dimensions()[1]]
        k1_ints = jnp.arange((len(k1)), dtype=int)
        k2_ints = jnp.arange((len(k2)), dtype=int)
        out_field = jnp.moveaxis(
            jnp.array(
                [[fn(k1_int, k2_int) for k2_int in k2_ints] for k1_int in k1_ints]
            ),
            -1,
            self.all_nonperiodic_dimensions()[0],
        )
        return FourierField(self.domain_no_hat, out_field, name=self.name + "_reconstr")


class FourierFieldSlice(FourierField):
    def __init__(
        self, domain, non_periodic_direction, field, name="field", *ks, **params
    ):
        self.domain = domain
        self.domain_no_hat = domain
        self.non_periodic_direction = non_periodic_direction
        self.field = field
        self.ks_raw = list(ks)
        self.ks = jnp.array(ks)
        self.ks_int = jnp.array(params["ks_int"])
        self.ks = jnp.insert(self.ks, non_periodic_direction, -1)
        self.ks_int = jnp.insert(self.ks_int, non_periodic_direction, -1)
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
        return [self.non_periodic_direction]

    def diff(self, direction, order=1):
        if direction in self.all_periodic_dimensions():
            out_field = (1j * self.ks[direction]) ** order * self.field
        else:
            out_field = self.domain.diff(self.field, 0, order)
        return FourierFieldSlice(
            self.domain_no_hat,
            self.non_periodic_direction,
            out_field,
            self.name + "_diff_" + str(order),
            *self.ks_raw,
            ks_int=self.ks_int
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
            *self.ks_raw,
            ks_int=self.ks_int
        )

    def assemble_poisson_matrix(self):
        y_mat = self.get_cheb_mat_2_homogeneous_dirichlet(0)
        n = y_mat.shape[0]
        factor = 0
        for direction in self.all_periodic_dimensions():
            factor += (1j * self.ks[direction]) ** 2

        I = jnp.eye(n)
        mat = factor * I + y_mat
        mat_inv = jnp.linalg.inv(mat)
        return mat_inv

    def solve_poisson(self, mat=None):
        if type(mat) == NoneType:
            mat_inv = self.assemble_poisson_matrix()
        else:
            k1 = self.ks_int[self.all_periodic_dimensions()[0]]
            k2 = self.ks_int[self.all_periodic_dimensions()[1]]
            mat_inv = mat[k1, k2, :, :]
        rhs_hat = self.field
        out_field = mat_inv @ rhs_hat
        out_fourier = FourierFieldSlice(
            self.domain,
            self.non_periodic_direction,
            out_field,
            self.name + "_poisson",
            *self.ks_raw,
            ks_int=self.ks_int[jnp.array(self.all_periodic_dimensions())]
        )
        return out_fourier

    def update_boundary_conditions(self):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        self.field = jnp.take(
            self.field,
            jnp.arange(len(self.domain.grid[0]))[1:-1],
            axis=0,
        )
        self.field = jnp.pad(
            self.field,
            [(1, 1)],
            mode="constant",
            constant_values=0.0,
        )

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
            *self.ks_raw,
            ks_int=self.ks_int
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
                *self.ks_raw,
                ks_int=self.ks_int
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
                *self.ks_raw,
                ks_int=self.ks_int
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
                *self.ks_raw,
                ks_int=self.ks_int
            )

    def shift(self, value):
        out_field = self.field + value
        return FourierField(self.domain_no_hat, out_field, name=self.name)

#!/usr/bin/env python3

from __future__ import annotations

import time

from abc import ABC, abstractmethod
import math
import jax
import jax.numpy as jnp
import matplotlib.figure as figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D # type: ignore
from matplotlib import colors
from scipy.interpolate import RegularGridInterpolator # type: ignore
import functools
import dataclasses
from typing import Callable, Iterable, Optional, Sequence, Union
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from pathlib import Path
import os

from jax_spectral_dns.domain import Domain, PhysicalDomain, FourierDomain

np_float_array = npt.NDArray[np.float64]
np_complex_array = npt.NDArray[np.complex64]
jnp_float_array = jnp.ndarray

NoneType = type(None)


class Field(ABC):
    """Class that holds the information needed to describe a dependent variable
    and to perform operations on it."""

    plotting_dir = "./plots/"
    plotting_format = ".png"

    # setting this to True activates increases performance dramatically,
    # though some convenience features such as printing or plotting intermediate
    # state are disabled.
    activate_jit_ = False

    field_dir = "./fields/"

    def __init__(self, domain: Domain, data: jnp.ndarray, name: str):
        self.domain = domain
        self.data = data
        self.name = name
        raise NotImplementedError("Trying to initialize an abstract class")

    @classmethod
    def Zeros(cls, domain, name="field") -> Self:
        data: jnp.ndarray = jnp.zeros(domain.shape)
        return cls(domain, data, name)

    @abstractmethod
    def get_domain(self) -> Domain:
        ...

    @abstractmethod
    def get_physical_domain(self) -> PhysicalDomain:
        ...

    @abstractmethod
    def normalize_by_max_value(self, return_max: bool=False) -> Union[Self, tuple[Self, jnp_float_array]]:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        ...

    @abstractmethod
    def update_boundary_conditions(self) -> None:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        ...

    @classmethod
    def initialize(cls, cleanup: bool=True) -> None:
        jax.config.update("jax_enable_x64", True) #type: ignore[no-untyped-call]
        newpaths = [Field.field_dir, Field.plotting_dir]
        for newpath in newpaths:
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        if cleanup:
            # clean plotting dir
            for file_ending in ["*.pdf", "*.png", "*.mp4"]:
                for f in Path(Field.plotting_dir).glob(file_ending):
                    if f.is_file():
                        f.unlink()

    def save_to_file(self, filename: str) -> None:
        """Save field to file filename."""
        field_array: np_float_array = np.array(self.data.tolist())
        field_array.dump(self.field_dir + filename)

    def get_data(self) -> jnp.ndarray:
        return self.data

    def get_name(self) -> str:
        """Return the name of the field."""
        return self.name

    def set_name(self, name: str) -> None:
        """Set the name of the field."""
        self.name = name

    def get_time_step(self) -> float:
        """Return the current time step of the field."""
        return self.time_step

    def set_time_step(self, time_step: float) -> None:
        """Set the current time step of the field."""
        self.time_step = time_step

    # @classmethod
    # def activate_jit(cls):
    #     cls.activate_jit_ = True

    def __add__(self: Self, _: Union[Self, jnp.ndarray]) -> Field:
        raise NotImplementedError()

    def __sub__(self, _: Union[Self, jnp.ndarray]) -> Field:
        raise NotImplementedError()

    def __mul__(self, _: Union[Self, jnp.ndarray, float]) -> Field:
        raise NotImplementedError()

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, _: float) -> Self:
        raise NotImplementedError()

    def __repr__(self):
        # self.plot()
        return str(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def max(self) -> jnp_float_array:
        return jnp.max(self.data.flatten())

    def min(self) -> jnp_float_array:
        return jnp.min(self.data.flatten())

    def absmax(self) -> jnp_float_array:
        max = jnp.max(self.data.flatten())
        min = jnp.min(self.data.flatten())
        return jnp.max(jnp.array([max, -min]))

    def __neg__(self):
        ret = self * (-1.0)
        ret.time_step = self.time_step
        return ret

    def __abs__(self):
        # TODO use integration or something more sophisticated
        return jnp.linalg.norm(self.data) / self.number_of_dofs()

    def energy(self) -> float:
        raise NotImplementedError()

    def normalize_by_energy(self) -> Self:
        en = self.energy()
        self.data = jax.lax.cond(en > 1e-20,
                                 lambda: self.data / en,
                                 lambda: self.data) # type: ignore[no-untyped-call]
        return self

    def number_of_dimensions(self) -> int:
        return len(self.all_dimensions())

    def number_of_dofs(self) -> int:
        return int(math.prod(self.get_domain().shape))

    def all_dimensions(self) -> Sequence[int]:
        return self.get_domain().all_dimensions()

    def is_periodic(self, direction) -> bool:
        return self.get_domain().is_periodic(direction)

    def all_periodic_dimensions(self) -> list[int]:
        return self.get_domain().all_periodic_dimensions()

    def all_nonperiodic_dimensions(self) -> list[int]:
        return self.get_domain().all_nonperiodic_dimensions()

    def pad_mat_with_zeros(self):
        return jnp.block(
            [
                [jnp.zeros((1, self.data.shape[1] + 2))],
                [
                    jnp.zeros((self.data.shape[0], 1)),
                    self.data,
                    jnp.zeros((self.data.shape[0], 1)),
                ],
                [jnp.zeros((1, self.data.shape[1] + 2))],
            ]
        )

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction: int) -> np_float_array:
        return self.get_physical_domain().get_cheb_mat_2_homogeneous_dirichlet(direction)

    @abstractmethod
    def diff(self, direction: int, order: int = 1) -> Field:
        ...

    def nabla(self):
        out = [self.diff(0)]
        out[0].name = "nabla_" + self.name + "_" + str(0)
        for dim in self.all_dimensions()[1:]:
            out.append(self.diff(dim))
            out[dim].name = "nabla_" + self.name + "_" + str(dim)
            out[dim].time_step = self.time_step
        return VectorField(out)

    def laplacian(self) -> Field:
        out: Field = self.diff(0, 2)
        for dim in self.all_dimensions()[1:]:
            assert isinstance(out, Field)
            out += self.diff(dim, 2)
        out.name = "lap_" + self.name
        out.time_step = self.time_step
        return out


class VectorField:
    def __init__(self, elements: Sequence[Field], name: Optional[str]=None):
        self.elements = elements
        self.name = name
        self.domain = elements[0].get_domain()

    @classmethod
    def Zeros(cls, field_cls, domain: Domain, name:str ="field") -> Self:
        dim = domain.number_of_dimensions
        return cls([field_cls.Zeros(domain) for _ in range(dim)], name)

    @classmethod
    def FromData(cls, field_cls, domain, data, name="field"):
        dim = domain.number_of_dimensions
        return cls([field_cls(domain, data[i]) for i in range(dim)])

    # def __getattr__(self, attr):
    #     def on_all(*args, **kwargs):
    #         acc = []
    #         for obj in self.elements:
    #             acc += [getattr(obj, attr)(*args, **kwargs)]
    #         return acc

    #     return on_all

    def __getitem__(self, index):
        return self.elements[index]

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        out = ""
        for elem in self.elements:
            out += str(elem)
        return out

    def __add__(self, other: Union[VectorField, jnp.ndarray]) -> VectorField:
        out = []
        for i in range(len(self)):
            out.append(self[i] + other[i])
        return VectorField(out)

    def __sub__(self, other: Union[VectorField, jnp.ndarray]) -> VectorField:
        return self + (-1) * other

    def __mul__(self, other: Union[VectorField, jnp.ndarray, float]) -> VectorField:
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

    def __truediv__(self, other: float) -> VectorField:
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

    def energy(self) -> float:
        en = 0
        for f in self:
            en += f.energy()
        return en

    def normalize_by_energy(self) -> Self:
        """Divide each field by the energy of the Vector field."""
        en = self.energy()
        for f in self:
            f.data = f.data / jnp.sqrt(en)
        return self

    def normalize_by_max_value(self, return_max: bool=False) -> Union[Self, tuple[Self, list[jnp_float_array]]]:
        """Divide each field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        max = []
        for f in self:
            if return_max:
                f, max_i = f.normalize_by_max_value(True)
                max.append(max_i)
            else:
                f, max_i = f.normalize_by_max_value(True)
        if return_max:
            return self, max
        else:
            return self

    def energy_norm(self, k):
        energy = k**2 * self[1] * self[1]
        energy += self[1].diff(1) * self[1].diff(1)
        vort = self.curl()
        energy += vort[1] * vort[1]
        return energy.volume_integral()

    def save_to_file(self, filename):
        """Save field to file filename.

        Note: the resulting format is compatible with dedalus. Importing e.g. a
        velocity field into dedalus is as straightforward as:
        u_array = np.load("/path/to/u_file", allow_pickle=True)
        v_array = np.load("/path/to/v_file", allow_pickle=True)
        w_array = np.load("/path/to/w_file", allow_pickle=True)

        ... # dedalus case setup goes here...
        u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
        u.data = np.stack([u_array, v_array, w_array])"""
        if self.name is None:
            self.set_name("field")
        else:
            self.set_name(self.name)
        for f in self:
            field_array = np.array(f.data.tolist())
            field_array.dump(f.field_dir + filename)

    def get_data(self):
        return jnp.array([f.data for f in self])
        # return [f.data for f in self]

    def get_time_step(self) -> int:
        time_steps = [f.time_step for f in self]
        out = max(time_steps)
        assert type(out) is int
        return out

    def set_time_step(self, time_step):
        self.time_step = time_step
        for j in range(len(self)):
            self[j].time_step = time_step

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        if len(self) == 1:
            assert type(self[0].name) is str
            return self[0].name
        else:
            assert type(self[0].name) is str
            name = self[0].name
            return name[:-2]

    def set_name(self, name: str) -> None:
        self.name = name
        for j in range(len(self)):
            self[j].name = name + "_" + "xyz"[j]

    def update_boundary_conditions(self):
        for field in self:
            field.update_boundary_conditions()

    def all_dimensions(self) -> Sequence[int]:
        out = self[0].get_domain().all_dimensions()
        assert type(out) is Sequence[int]
        return out

    def all_periodic_dimensions(self) -> list[int]:
        out = self[0].get_domain().all_periodic_dimensions()
        assert type(out) is list[int]
        return out

    def all_nonperiodic_dimensions(self) -> list[int]:
        out = self[0].get_domain().all_nonperiodic_dimensions()
        assert type(out) is list[int]
        return out

    def hat(self) -> VectorField:
        return VectorField([f.hat() for f in self])

    def no_hat(self) -> VectorField:
        return VectorField([f.no_hat() for f in self])

    def plot(self, *other_fields):
        for i in jnp.arange(len(self)):
            other_fields_i = [item[i] for item in other_fields]
            self[i].plot(*other_fields_i)

    def plot_3d(self, direction: Optional[int]=None) -> None:
        for f in self:
            assert type(f) is PhysicalField, "plot_3d only defined for PhysicalField."
            f.plot_3d(direction)

    def cross_product(self, other: VectorField) -> VectorField:
        out_0 = self[1] * other[2] - self[2] * other[1]
        out_1 = self[2] * other[0] - self[0] * other[2]
        out_2 = self[0] * other[1] - self[1] * other[0]

        time_step = self.get_time_step()
        out = [out_0, out_1, out_2]
        for f in out:
            f.time_step = time_step

        return VectorField(out)

    def curl(self) -> VectorField:
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
        if self.name is not None:
            out.name = "div_" + self.name + "_" + str(0)
        else:
            out.name = "div_field"
        for dim in self.all_dimensions()[1:]:
            out += self[dim].diff(dim)
        return out

    def reconstruct_from_wavenumbers(self, fn, number_of_other_fields=0):

        # jit = True
        # vectorize = True

        jit = False
        # vectorize = jax.devices()[0].platform == "gpu"  # True on GPUs and False on CPUs
        vectorize = False

        if jit:
            raise NotImplementedError()
            # time_1 = time.time()
            # out_field = reconstruct_from_wavenumbers_jit(self[0].domain_no_hat, fn)
            # time_2 = time.time()

            # out = VectorField(
            #     [
            #         FourierField(self[0].domain_no_hat, out_field[i])
            #         for i in self.all_dimensions()
            #     ]
            # )
            # time_3 = time.time()
            # print("in reconstr: time for part 1: ", time_2 - time_1)
            # print("in reconstr: time for part 2: ", time_3 - time_2)
            # # out = VectorField(jax.lax.map(lambda i: FourierField(self[0].domain_no_hat, out_field.at[i].get()), jnp.arange(self[0].domain.number_of_dimensions)))
            # return out
        else:
            time_1 = time.time()
            k1s = jnp.array(
                self[0]
                .fourier_domain.grid[self.all_periodic_dimensions()[0]]
                .astype(int)
            )
            k2s = jnp.array(
                self[0]
                .fourier_domain.grid[self.all_periodic_dimensions()[1]]
                .astype(int)
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
                        self[0].physical_domain,
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
                            self[0].physical_domain,
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
                        self[0].physical_domain,
                        jnp.moveaxis(
                            out_array[i, :, :].reshape(Nx, Nz, Ny),
                            -1,
                            self.all_nonperiodic_dimensions()[0],
                        ),
                        "other_" + str(i),
                    )
                    for i in self.all_dimensions()
                ]
            return (VectorField(out_field), 0)

    def plot_streamlines(self, normal_direction):
        if not self[0].activate_jit_:
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            directions = [i for i in self.all_dimensions() if i != normal_direction]
            x = self.domain.grid[directions[0]]
            y = jnp.flip(self.domain.grid[directions[1]])
            N = 40
            xi = np.linspace(x[0], x[-1], N)
            yi = np.linspace(y[0], y[-1], N)
            N_c = self.domain.number_of_cells(normal_direction) // 2
            U = self[directions[0]].data.take(indices=N_c, axis=normal_direction)
            V = self[directions[1]].data.take(indices=N_c, axis=normal_direction)
            interp_u = RegularGridInterpolator((x, y), U, method="cubic")
            interp_v = RegularGridInterpolator((x, y), V, method="cubic")
            Ui = np.array([[interp_u([[x_, y_]])[0] for x_ in xi] for y_ in yi])
            Vi = np.array([[interp_v([[x_, y_]])[0] for x_ in xi] for y_ in yi])

            try:
                ax.streamplot(xi, yi, Ui, Vi, broken_streamlines=False, linewidth=0.4)
            except TypeError:  # compatibilty with older matplotlib versions
                ax.streamplot(xi, yi, Ui, Vi, linewidth=0.4)
            def save() -> None:
                fig.savefig(
                    self[0].plotting_dir
                    + "plot_streamlines_"
                    + self.get_name()
                    + "_t_"
                    + "{:06}".format(self[0].time_step)
                    + self[0].plotting_format
                )
                fig.savefig(
                    self[0].plotting_dir
                    + "plot_streamlines_"
                    + self.get_name()
                    + "_latest"
                    + self[0].plotting_format
                )
            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()

    def plot_vectors(self, normal_direction):
        if not self[0].activate_jit_:
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            directions = [i for i in self.all_dimensions() if i != normal_direction]
            x = self.domain.grid[directions[0]]
            y = jnp.flip(self.domain.grid[directions[1]])
            N = 40
            xi = np.linspace(x[0], x[-1], N)
            yi = np.linspace(y[0], y[-1], N)
            N_c = self.domain.number_of_cells(normal_direction) // 2
            U = self[directions[0]].data.take(indices=N_c, axis=normal_direction)
            V = self[directions[1]].data.take(indices=N_c, axis=normal_direction)
            interp_u = RegularGridInterpolator((x, y), U, method="cubic")
            interp_v = RegularGridInterpolator((x, y), V, method="cubic")
            Ui = np.array([[interp_u([[x_, y_]])[0] for x_ in xi] for y_ in yi])
            Vi = np.array([[interp_v([[x_, y_]])[0] for x_ in xi] for y_ in yi])

            ax.quiver(xi, yi, Ui, Vi)
            def save() -> None:
                fig.savefig(
                    self[0].plotting_dir
                    + "plot_vectors_"
                    + self.get_name()
                    + "_t_"
                    + "{:06}".format(self[0].time_step)
                    + self[0].plotting_format
                )
                fig.savefig(
                    self[0].plotting_dir
                    + "plot_vectors_"
                    + self.get_name()
                    + "_latest"
                    + self[0].plotting_format
                )
            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()


# @register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=False)
class PhysicalField(Field):
    # def tree_flatten(self):
    #     children = (self.data, self.time_step)
    #     aux_data = (self.physical_domain, self.name)
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(aux_data[0], children[0], aux_data[1], children[2])

    def __init__(
        self,
        domain: PhysicalDomain,
        data: jnp.ndarray,
        name: str = "field",
        time_step: int = 0,
    ):
        self.physical_domain = domain
        self.data = data
        self.name = name
        self.time_step: int = time_step

    def shift(self, value):
        out_field = self.data + value
        return PhysicalField(self.physical_domain, out_field, name=self.name)

    def __add__(self, other: Union[Self, jnp.ndarray]) -> PhysicalField:
        assert not isinstance(
            other, FourierField
        ), "Attempted to add a Field and a Fourier Field."
        if isinstance(other, PhysicalField):
            # if True: # TODO
            if self.activate_jit_:
                new_name = ""
            else:
                assert isinstance(other, PhysicalField)
                if other.name[0] == "-":
                    new_name = self.name + " - " + other.name[1:]
                else:
                    new_name = self.name + " + " + other.name
            ret = PhysicalField(
                self.physical_domain, self.data + other.data, name=new_name
            )
        else:
            print(type(other))
            assert isinstance(other, jnp.ndarray)
            ret = PhysicalField(self.physical_domain, self.data + other, name="field")
        ret.time_step = self.time_step
        return ret 

    def __sub__(self, other: Union[Self, jnp.ndarray]) -> PhysicalField:
        assert not isinstance(
            other, FourierField
        ), "Attempted to subtract a Field and a Fourier Field."
        return self + other * (-1.0) # type: ignore

    def __mul__(self, other: Union[Self, jnp.ndarray, float]) -> PhysicalField:
        if isinstance(other, FourierField):
            raise Exception("Attempted to multiply physical field and Fourier field")
        elif isinstance(other, Field):
            if self.activate_jit_:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            ret = PhysicalField(
                self.physical_domain, self.data * other.data, name=new_name
            )
            ret.time_step = self.time_step
            return ret 
        else:
            if self.activate_jit_:
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
            ret = PhysicalField(self.physical_domain, self.data * other, name=new_name)
            ret.time_step = self.time_step
            return ret

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(
        self, other: float
    ) -> PhysicalField:
        if isinstance(other, Field):
            raise Exception("Don't know how to divide by another field")
        else:
            new_name = "field"
            ret = PhysicalField(self.physical_domain, self.data * other, name=new_name)
            ret.time_step = self.time_step
            return ret

    @classmethod
    def FromFunc(cls, domain: PhysicalDomain, func: Optional[Callable[[Sequence[float]], float]]=None, name: str="field") -> Self:
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
        zero_field = PhysicalField.FromFunc(domain)
        rands = []
        for i in jnp.arange(zero_field.number_of_dofs()):
            key, subkey = jax.random.split(key)
            rands.append(
                jax.random.uniform(subkey, minval=interval[0], maxval=interval[1])
            )
        field = jnp.array(rands).reshape(zero_field.physical_domain.shape)
        return cls(domain, field, name)

    @classmethod
    def FromField(cls, domain: PhysicalDomain, field: PhysicalField) -> PhysicalField:
        """Construct a new field depending on the independent variables described by
        domain by interpolating a given field."""
        # TODO testing + performance improvements needed
        assert not isinstance(
            field, FourierField
        ), "Attempted to interpolate a Field from a FourierField."
        out = []
        if domain.number_of_dimensions == 1:
            for x in domain.grid[0]:
                out.append(field.eval(x))
                out_array = jnp.array(out)
        elif domain.number_of_dimensions == 2:
            for x in domain.grid[0]:
                for y in domain.grid[1]:
                    out.append(field.eval([x, y]))
            out_array = jnp.array(out)
            out_array.reshape(domain.number_of_cells(0), domain.number_of_cells(1))
        elif domain.number_of_dimensions == 3:
            for x in domain.grid[0]:
                for y in domain.grid[1]:
                    for z in domain.grid[2]:
                        out.append(field.eval([x, y, z]))
            out_arr = jnp.array(out)
            out_arr.reshape(
                domain.number_of_cells(0),
                domain.number_of_cells(1),
                domain.number_of_cells(2),
            )
        else:
            raise NotImplementedError("Number of dimensions not supported.")
        return PhysicalField(domain, out_arr, field.name + "_projected")

    @classmethod
    def FromFile(cls, domain: PhysicalDomain, filename: str, name: str="field") -> PhysicalField:
        """Construct new field depending on the independent variables described
        by domain by reading in a saved field from file filename."""
        out = PhysicalField.Zeros(domain, name=name)
        field_array = np.load(out.field_dir + filename, allow_pickle=True)
        out.data = jnp.array(field_array.tolist())
        return out

    def normalize_by_max_value(self, return_max: bool=False) -> Union[Self, tuple[Self, jnp_float_array]]:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        max = abs(self.absmax())

        self.data = jax.lax.cond(
            max > 1e-20,
            lambda: self.data / max,
            lambda: self.data

        ) # type: ignore[no-untyped-call]
        if return_max:
            return self, max
        else:
            return self

    def get_domain(self) -> PhysicalDomain:
        return self.physical_domain

    def get_physical_domain(self) -> PhysicalDomain:
        return self.get_domain()

    def l2error(self, fn):
        # TODO supersampling
        analytical_solution = PhysicalField.FromFunc(self.physical_domain, fn)
        return jnp.linalg.norm((self - analytical_solution).data, None)

    def volume_integral(self) -> float:
        int = PhysicalField(self.physical_domain, self.data)
        dims = list(reversed(self.all_dimensions()))
        for i in dims[:-1]:
            assert type(int) is PhysicalField
            out_ = int.definite_integral(i)
            assert type(out_) is PhysicalField
            int = out_
        assert type(int) is PhysicalField
        out = int.definite_integral(dims[-1])
        assert type(out) is float
        return out

    def energy(self):
        energy = 0.5 * self * self
        domain_volume = 2.0 ** (len(self.all_nonperiodic_dimensions())) * jnp.prod(
            jnp.array(self.physical_domain.scale_factors)
        )  # nonperiodic dimensions are size 2, but its scale factor is only 1
        return energy.volume_integral() / domain_volume

    def update_boundary_conditions(self):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        for dim in self.all_nonperiodic_dimensions():
            self.data = jnp.take(
                self.data,
                jnp.arange(self.physical_domain.number_of_cells(dim))[1:-1],
                axis=dim,
            )
            self.data = jnp.pad(
                self.data,
                [
                    (0, 0) if self.physical_domain.periodic_directions[d] else (1, 1)
                    for d in self.all_dimensions()
                ],
                mode="constant",
                constant_values=0.0,
            )

    def eval(self, X: Sequence[float]) -> jnp_float_array:
        """Evaluate field at arbitrary point X through linear interpolation. (TODO: This could obviously be improved for Chebyshev dirctions, but this is not yet implemented)"""
        grd = self.physical_domain.grid
        interpolant = []
        weights = []
        for dim in self.all_dimensions():
            i = jnp.argmin(
                (grd[dim] - X[dim])[:-1] * (jnp.roll(grd[dim], -1) - X[dim])[:-1]
            )
            interpolant.append(i)
            weights.append((grd[dim][i] - X[dim]) / (grd[dim][i] - grd[dim][i + 1]))
        base_value = self.data
        other_values = []
        for dim in self.all_dimensions():
            other_values.append(self.data)
        for dim in reversed(self.all_dimensions()):
            base_value = jnp.take(base_value, indices=interpolant[dim], axis=dim)
            for j in self.all_dimensions():
                if j == dim:
                    other_values[j] = jnp.take(
                        other_values[j], indices=interpolant[dim] + 1, axis=dim
                    )
                else:
                    other_values[j] = jnp.take(
                        other_values[j], indices=interpolant[dim], axis=dim
                    )
        out = base_value
        for dim in self.all_dimensions():
            out += (other_values[dim] - base_value) * weights[dim]
        return out

    def plot_center(self, dimension, *other_fields):
        if not self.activate_jit_:
            if self.physical_domain.number_of_dimensions == 1:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                ax.plot(self.physical_domain.grid[0], self.data, label=self.name)
                for other_field in other_fields:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        other_field.data,
                        "--",
                        label=other_field.name,
                    )
                fig.legend()
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
                # plt.close(fig)
            elif self.physical_domain.number_of_dimensions == 2:
                # fig, ax = plt.subplots(1, 1)
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                other_dim = [i for i in self.all_dimensions() if i != dimension][0]
                N_c = self.physical_domain.number_of_cells(other_dim) // 2
                ax.plot(
                    self.physical_domain.grid[dimension],
                    self.data.take(indices=N_c, axis=other_dim),
                    label=self.name,
                )
                for other_field in other_fields:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        other_field.data.take(indices=N_c, axis=other_dim),
                        "--",
                        label=other_field.name,
                    )
                fig.legend()
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
                # plt.close(fig)
            elif self.physical_domain.number_of_dimensions == 3:
                # fig, ax = plt.subplots(1, 1)
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                other_dims = [i for i in self.all_dimensions() if i != dimension]
                N_cs = [
                    self.physical_domain.number_of_cells(dim) // 2 for dim in other_dims
                ]
                ax.plot(
                    self.physical_domain.grid[dimension],
                    self.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                        indices=N_cs[0], axis=other_dims[0]
                    ),
                    label=self.name,
                )
                for other_field in other_fields:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        other_field.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                            indices=N_cs[0], axis=other_dims[0]
                        ),
                        "--",
                        label=other_field.name,
                    )
                fig.legend()
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
            else:
                raise Exception("Not implemented yet")

    def plot(self, *other_fields):
        if not self.activate_jit_:
            if self.physical_domain.number_of_dimensions == 1:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                ax.plot(
                    self.physical_domain.grid[0],
                    self.data,
                    label=self.name,
                )
                for other_field in other_fields:
                    ax.plot(
                        self.physical_domain.grid[0],
                        other_field.data,
                        "--",
                        label=other_field.name,
                    )
                fig.legend()
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()

            elif self.physical_domain.number_of_dimensions == 2:
                fig = figure.Figure(figsize=(15, 5))
                ax = np.array([fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)])
                assert type(ax) is np.ndarray
                ax3d = fig.add_subplot(1, 3, 3, projection="3d")
                assert type(ax3d) is Axes3D
                for dimension in self.all_dimensions():
                    other_dim = [i for i in self.all_dimensions() if i != dimension][0]
                    N_c = self.physical_domain.number_of_cells(other_dim) // 2
                    ax[dimension].plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_c, axis=other_dim),
                        label=self.name,
                    )
                    ax3d.plot_surface(
                        self.physical_domain.mgrid[0],
                        (self.physical_domain.mgrid[1]),
                        self.data,
                    )
                    for other_field in other_fields:
                        ax[dimension].plot(
                            self.physical_domain.grid[dimension],
                            other_field.data.take(indices=N_c, axis=other_dim),
                            "--",
                            label=other_field.name,
                        )
                        ax3d.plot_surface(
                            self.physical_domain.mgrid[0],
                            (self.physical_domain.mgrid[1]),
                            other_field.data,
                        )
                    fig.legend()
                    def save() -> None:
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
                    try:
                        save()
                    except FileNotFoundError:
                        Field.initialize(False)
                        save()
            elif self.physical_domain.number_of_dimensions == 3:
                fig = figure.Figure()
                # ax = fig.subplots(1, 3, figsize=(15, 5))
                ax = fig.subplots(1, 3)
                assert type(ax) is np.ndarray
                for dimension in self.all_dimensions():
                    other_dims = [i for i in self.all_dimensions() if i != dimension]
                    N_cs = [
                        self.physical_domain.number_of_cells(dim) // 2
                        for dim in other_dims
                    ]
                    ax[dimension].plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                            indices=N_cs[0], axis=other_dims[0]
                        ),
                        label=self.name,
                    )
                    for other_field in other_fields:
                        ax[dimension].plot(
                            self.physical_domain.grid[dimension],
                            other_field.data.take(
                                indices=N_cs[1], axis=other_dims[1]
                            ).take(indices=N_cs[0], axis=other_dims[0]),
                            "--",
                            label=other_field.name,
                        )
                fig.legend()
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
            else:
                raise Exception("Not implemented yet")

    def plot_3d(self, direction: Optional[int]=None) -> None:
        if not self.activate_jit_:
            if direction is not None:
                self.plot_3d_single(direction)
            else:
                assert (
                    self.physical_domain.number_of_dimensions == 3
                ), "Only 3D supported for this plotting method."
                fig = figure.Figure(layout="constrained")
                base_len = 100
                grd = (base_len, base_len)
                lx = self.physical_domain.scale_factors[0]
                ly = self.physical_domain.scale_factors[1] * 2
                lz = self.physical_domain.scale_factors[2]
                rows_x = int(ly / (ly + lx) * base_len)
                cols_x = int(lz / (lz + ly) * base_len)
                rows_y = int(lx / (ly + lx) * base_len)
                cols_y = int(lz / (lz + ly) * base_len)
                rows_z = int(lx / (ly + lx) * base_len)
                cols_z = int(ly / (lz + ly) * base_len)
                ax = [
                    fig.add_subplot(
                        fig.add_gridspec(*grd)[0 : 0 + rows_x, 0 : 0 + cols_x]
                    ),
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
                    N_c = self.physical_domain.number_of_cells(dim) // 2
                    other_dim = [i for i in self.all_dimensions() if i != dim]
                    ims.append(
                        ax[dim].imshow(
                            self.data.take(indices=N_c, axis=dim),
                            interpolation=None,
                            extent=(
                                self.physical_domain.grid[other_dim[1]][0],
                                self.physical_domain.grid[other_dim[1]][-1],
                                self.physical_domain.grid[other_dim[0]][0],
                                self.physical_domain.grid[other_dim[0]][-1],
                            ),
                        )
                    )
                    ax[dim].set_xlabel("xyz"[other_dim[1]])
                    ax[dim].set_ylabel("xyz"[other_dim[0]])
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in ims) # type: ignore[union-attr]
                vmax = max(image.get_array().max() for image in ims) # type: ignore[union-attr]
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in ims:
                    im.set_norm(norm)
                fig.colorbar(ims[0], ax=ax, label=self.name)
                def save() -> None:
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
                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()

    def plot_3d_single(self, dim: int) -> None:
        if not self.activate_jit_:
            assert (
                self.physical_domain.number_of_dimensions == 3
            ), "Only 3D supported for this plotting method."
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ims = []
            N_c = self.physical_domain.number_of_cells(dim) // 2
            other_dim = [i for i in self.all_dimensions() if i != dim]
            ims.append(
                ax.imshow(
                    self.data.take(indices=N_c, axis=dim).T,
                    interpolation=None,
                    extent=(
                        self.physical_domain.grid[other_dim[0]][0],
                        self.physical_domain.grid[other_dim[0]][-1],
                        self.physical_domain.grid[other_dim[1]][0],
                        self.physical_domain.grid[other_dim[1]][-1],
                    ),
                )
            )
            ax.set_xlabel("xyz"[other_dim[0]])
            ax.set_ylabel("xyz"[other_dim[1]])
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in ims) # type: ignore[union-attr]
            vmax = max(image.get_array().max() for image in ims) # type: ignore[union-attr]
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in ims:
                im.set_norm(norm)
            fig.colorbar(ims[0], ax=ax, label=self.name, orientation="vertical")
            def save() -> None:
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
            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()

    def plot_isolines(self, normal_direction, isolines=None):
        if not self.activate_jit_:
            if type(isolines) == NoneType:
                isolines = [0, 1.5, 2.5, 3.5]
                isolines += [-i for i in isolines[1:]]

            assert isolines is not None
            isolines.sort()

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            directions = [i for i in self.all_dimensions() if i != normal_direction]
            x = self.physical_domain.grid[directions[0]]
            y = self.physical_domain.grid[directions[1]]
            X, Y = jnp.meshgrid(x, y)
            N_c = self.physical_domain.number_of_cells(normal_direction) // 2
            f = self.data.take(indices=N_c, axis=normal_direction).T
            cmap = colors.ListedColormap((("gray", 0.3), "white")) # type: ignore[arg-type]
            bounds = [-1e10, 0, 1e10]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            ax.imshow(
                f,
                interpolation="gaussian",
                origin="lower",
                cmap=cmap,
                norm=norm,
                extent=(x[0], x[-1], y[0], y[-1]),
            )
            CS = ax.contour(X, Y, f, isolines)
            ax.clabel(CS, inline=True, fontsize=10)
            def save() -> None:
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
            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()

    def hat(self):
        out = FourierField.FromField(self)
        out.time_step = self.time_step
        return out

    def diff(self, direction, order=1):
        name_suffix = "".join([["x", "y", "z"][direction] for _ in jnp.arange(order)])
        return PhysicalField(
            self.physical_domain,
            self.physical_domain.diff(self.data, direction, order),
            self.name + "_" + name_suffix,
        )

    def integrate(self, direction: int, order: int=1, bc_left: Optional[float]=None, bc_right: Optional[float]=None) -> PhysicalField:
        out_bc = self.physical_domain.integrate(
            self.data, direction, order, bc_left, bc_right
        )
        assert type(out_bc) is jnp_float_array
        return PhysicalField(self.physical_domain, out_bc, name=self.name + "_int")

    def definite_integral(
        self, direction: int
    ) -> Union[float, jnp_float_array, PhysicalField]:
        def reduce_add_along_axis(arr: jnp_float_array, axis: int) -> jnp_float_array:
            # return np.add.reduce(arr, axis=axis)
            arr = jnp.moveaxis(arr, axis, 0)
            out_arr = functools.reduce(lambda a, b: a + b, arr)
            assert type(out_arr) is jnp_float_array
            return out_arr

        if not self.is_periodic(direction):
            int = self.integrate(direction, 1, bc_right=0.0)
            if self.number_of_dimensions() == 1:
                out = int[0] - int[-1]
                assert type(out) is float
                return out
            else:
                N = self.physical_domain.number_of_cells(direction)
                inds = [i for i in self.all_dimensions() if i != direction]
                shape = tuple(
                    (np.array(self.physical_domain.shape)[tuple(inds),]).tolist()
                )
                periodic_directions = tuple(
                    (
                        np.array(self.physical_domain.periodic_directions)[
                            tuple(inds),
                        ]
                    ).tolist()
                )
                scale_factors = tuple(
                    (
                        np.array(self.physical_domain.scale_factors)[tuple(inds),]
                    ).tolist()
                )
                reduced_domain = PhysicalDomain.create(
                    shape, periodic_directions, scale_factors=scale_factors, aliasing=self.get_domain().aliasing
                )
                field = jnp.take(int.data, indices=0, axis=direction) - jnp.take(
                    int.data, indices=N - 1, axis=direction
                )
                return PhysicalField(reduced_domain, field)
        else:
            N = self.physical_domain.number_of_cells(direction)
            if self.number_of_dimensions() == 1:
                return (
                    self.physical_domain.scale_factors[direction]
                    / N
                    * jnp.sum(self.data[:])
                )
            else:
                N = self.physical_domain.number_of_cells(direction)
                inds = [i for i in self.all_dimensions() if i != direction]
                shape = tuple(
                    (np.array(self.physical_domain.shape)[tuple(inds),]).tolist()
                )
                periodic_directions = tuple(
                    (
                        np.array(self.physical_domain.periodic_directions)[
                            tuple(inds),
                        ]
                    ).tolist()
                )
                scale_factors = tuple(
                    (
                        np.array(self.physical_domain.scale_factors)[tuple(inds),]
                    ).tolist()
                )
                reduced_domain = PhysicalDomain.create(
                    shape, periodic_directions, scale_factors=scale_factors, aliasing=self.get_domain().aliasing
                )
                field = (
                    self.physical_domain.scale_factors[direction]
                    / N
                    * reduce_add_along_axis(self.data, direction)
                )
                return PhysicalField(reduced_domain, field)


# @register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=False)
class FourierField(Field):
    # def tree_flatten(self):
    #     children = (self.data, self.time_step)
    #     aux_data = (self.physical_domain, self.name, self.fourier_domain)
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(aux_data[0], children[0], aux_data[1], children[1], aux_data[2])

    def __init__(
        self,
        domain: PhysicalDomain,
        data: jnp.ndarray,
        name: str = "field_hat",
        time_step: int = 0,
        fourier_domain: Optional[FourierDomain] = None,
    ):
        self.name = name
        self.time_step: int = time_step
        self.physical_domain = domain
        if fourier_domain is None:
            self.fourier_domain = domain.hat()
        else:
            assert type(fourier_domain) is FourierDomain
            self.fourier_domain = fourier_domain
        self.data = data

    @classmethod
    def FromRandom(cls, domain: PhysicalDomain, seed: int=0, interval: tuple[float, float]=(-0.1, 0.1), name:str ="field") -> FourierField:
        """Construct a random field depending on the independent variables described by domain."""
        # TODO generate "nice" random fields
        key = jax.random.PRNGKey(seed)
        zero_field = PhysicalField.FromFunc(domain)
        rands = []
        for _ in jnp.arange(zero_field.number_of_dofs()):
            key, subkey = jax.random.split(key)
            rands.append(
                jax.random.uniform(subkey, minval=interval[0], maxval=interval[1])
            )
        field = jnp.array(rands).reshape(zero_field.physical_domain.shape)
        return cls(domain, field, name)

    def get_domain(self) -> FourierDomain:
        return self.fourier_domain

    def get_physical_domain(self) -> PhysicalDomain:
        return self.physical_domain

    def __add__(self, other: Union[Self, jnp.ndarray]) -> FourierField:
        assert isinstance(
            other, FourierField
        ), "Attempted to add a Fourier Field and a Field."
        if self.activate_jit_:
            new_name = ""
        else:
            if other.name[0] == "-":
                new_name = self.name + " - " + other.name[1:]
            else:
                new_name = self.name + " + " + other.name
        ret = FourierField(self.physical_domain, self.data + other.data, name=new_name)
        ret.time_step = self.time_step
        return ret

    def __sub__(self, other: Union[Self, jnp.ndarray]) -> FourierField:
        return self + other * (-1.0) # type: ignore

    def __mul__(self, other: Union[Self, jnp.ndarray, float]) -> FourierField:
        if isinstance(other, Field):
            assert isinstance(
                other, FourierField
            ), "Attempted to multiply a Fourier Field and a Field."

        if isinstance(other, FourierField):
            if self.activate_jit_:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            ret = FourierField(
                self.physical_domain, self.data * other.data, name=new_name
            )
            ret.time_step = self.time_step
            return ret
        else:
            if self.activate_jit_:
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
            ret = FourierField(self.physical_domain, self.data * other, name=new_name)
            ret.time_step = self.time_step
            return ret

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other: float) -> FourierField:
        out = super().__truediv__(other)
        return FourierField(self.physical_domain, out.data, name=out.name)

    @classmethod
    def FromField(cls, field: PhysicalField) -> FourierField:
        out = cls(field.physical_domain, field.data, field.name + "_hat")
        out.physical_domain = field.physical_domain
        out.fourier_domain = field.physical_domain.hat()
        out.data = out.physical_domain.field_hat(field.data)
        return out

    def normalize_by_max_value(self, return_max: bool=False) -> Union[Self, tuple[Self, jnp_float_array]]:
        raise Exception("This is not supported for Fourier Fields. Transform to PhysicalField, normalize, and transform back to FourierField instead.")


    def diff(self, direction, order=1):
        if direction in self.all_periodic_dimensions():
            out_field = (1j * self.fourier_domain.mgrid[direction]) ** order * self.data
        else:
            out_field = self.physical_domain.diff(self.data, direction, order)
        return FourierField(
            self.physical_domain,
            out_field,
            name=self.name + "_diff_" + str(order),
        )

    def integrate(self, direction, order=1, bc_right=None, bc_left=None):
        if direction in self.all_periodic_dimensions():
            mgrid = self.fourier_domain.mgrid[direction]
            field = self.data
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
            out_field_ = (
                self.physical_domain
                .integrate(self.data, direction, order, bc_right=bc_right, bc_left=bc_left)
            )
            assert type(out_field_) is jnp_float_array
            out_field = out_field_

        return FourierField(
            self.physical_domain, out_field, name=self.name + "_int_" + str(order)
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

    def update_boundary_conditions(self) -> None:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        raise NotImplementedError()

    def assemble_poisson_matrix(self) -> np_complex_array:
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
        k1 = self.fourier_domain.grid[self.all_periodic_dimensions()[0]]
        k2 = self.fourier_domain.grid[self.all_periodic_dimensions()[1]]
        k1sq = k1**2
        k2sq = k2**2
        mat = np.array(
            [
                [np.linalg.inv((-(k1sq_ + k2sq_)) * eye_bc + y_mat) for k2sq_ in k2sq]
                for k1sq_ in k1sq
            ]
        )
        return mat

    def solve_poisson(self, mat=None):
        assert len(self.all_dimensions()) == 3, "Only 3d implemented currently."
        assert (
            len(self.all_nonperiodic_dimensions()) <= 1
        ), "Poisson solution not implemented for the general case."
        rhs_hat = self.data
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
            self.physical_domain, out_field, name=self.name + "_poisson"
        )
        return out_fourier

    def no_hat(self):
        out = self.fourier_domain.no_hat(self.data)
        out_field = PhysicalField(
            self.physical_domain, out, name=(self.name).replace("_hat", "")
        )
        out_field.time_step = self.time_step
        return out_field

    def reconstruct_from_wavenumbers(self, fn, vectorize=False):
        if vectorize:
            print("vectorisation not implemented yet, using unvectorized version")
        assert self.number_of_dimensions() != 3, "2D not implemented yet"
        k1 = self.fourier_domain.grid[self.all_periodic_dimensions()[0]]
        k2 = self.fourier_domain.grid[self.all_periodic_dimensions()[1]]
        k1_ints = jnp.arange((len(k1)), dtype=int)
        k2_ints = jnp.arange((len(k2)), dtype=int)
        out_field = jnp.moveaxis(
            jnp.array(
                [[fn(k1_int, k2_int) for k2_int in k2_ints] for k1_int in k1_ints]
            ),
            -1,
            self.all_nonperiodic_dimensions()[0],
        )
        return FourierField(
            self.physical_domain, out_field, name=self.name + "_reconstr"
        )


class FourierFieldSlice(FourierField):
    def __init__(
        self, domain, non_periodic_direction, data, name: str="field_hat_slice", *ks, **params
    ):
        # self.physical_domain = domain
        self.fourier_domain = domain
        self.non_periodic_direction = non_periodic_direction
        self.data = data
        self.ks_raw = list(ks)
        self.ks = jnp.array(ks)
        self.ks_int = jnp.array(params["ks_int"])
        self.ks = jnp.insert(self.ks, non_periodic_direction, -1)
        self.ks_int = jnp.insert(self.ks_int, non_periodic_direction, -1)
        self.name = name

    def all_dimensions(self) -> Sequence[int]:
        return range(len(self.ks))

    def all_periodic_dimensions(self) -> list[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if d not in self.all_nonperiodic_dimensions()
        ]

    def all_nonperiodic_dimensions(self) -> list[int]:
        return [self.non_periodic_direction]

    def diff(self, direction, order=1):
        if direction in self.all_periodic_dimensions():
            out_field = (1j * self.ks[direction]) ** order * self.data
        else:
            out_field = self.physical_domain.diff(self.data, 0, order)
        return FourierFieldSlice(
            self.fourier_domain,
            self.non_periodic_direction,
            out_field,
            self.name + "_diff_" + str(order),
            *self.ks_raw,
            ks_int=self.ks_int,
        )

    def integrate(self, direction, order=1, bc_right=None, bc_left=None):
        if direction in self.all_periodic_dimensions():
            out_field = self.data / (1j * self.ks[direction]) ** order
        else:
            out_field = self.fourier_domain.integrate(self.data, 0, order)
        return FourierFieldSlice(
            self.fourier_domain,
            self.non_periodic_direction,
            out_field,
            self.name + "_int_" + str(order),
            *self.ks_raw,
            ks_int=self.ks_int,
        )

    def assemble_poisson_matrix(self) -> np_complex_array:
        y_mat = self.get_cheb_mat_2_homogeneous_dirichlet(0)
        n = y_mat.shape[0]
        factor = np.zeros_like(self.ks[0])
        for direction in self.all_periodic_dimensions():
            factor += (1j * self.ks[direction]) ** 2

        I = np.eye(n)
        mat = factor * I + y_mat
        mat_inv = np.linalg.inv(mat)
        assert type(mat_inv) is np_complex_array
        return mat_inv

    def solve_poisson(self, mat=None):
        if type(mat) == NoneType:
            mat_inv = self.assemble_poisson_matrix()
        else:
            k1 = self.ks_int[self.all_periodic_dimensions()[0]]
            k2 = self.ks_int[self.all_periodic_dimensions()[1]]
            mat_inv = mat[k1, k2, :, :]
        rhs_hat = self.data
        out_field = mat_inv @ rhs_hat
        out_fourier = FourierFieldSlice(
            self.fourier_domain,
            self.non_periodic_direction,
            out_field,
            self.name + "_poisson",
            *self.ks_raw,
            ks_int=self.ks_int[jnp.array(self.all_periodic_dimensions())],
        )
        return out_fourier

    def update_boundary_conditions(self):
        """This assumes homogeneous dirichlet conditions in all non-periodic directions"""
        self.data = jnp.take(
            self.data,
            jnp.arange(len(self.fourier_domain.grid[0]))[1:-1],
            axis=0,
        )
        self.data = jnp.pad(
            self.data,
            [(1, 1)],
            mode="constant",
            constant_values=0.0,
        )

    def __neg__(self) -> FourierFieldSlice:
        return self * (-1.0)

    def __add__(self, other: Union[Self, jnp.ndarray]) -> FourierFieldSlice:
        if isinstance(other, FourierFieldSlice):
            if self.activate_jit_:
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
                self.fourier_domain,
                self.non_periodic_direction,
                self.data + other.data,
                new_name,
                *self.ks_raw,
                ks_int=self.ks_int,
            )
        else:
            return FourierFieldSlice(
                self.fourier_domain,
                self.non_periodic_direction,
                self.data + other,
                self.name,
                *self.ks_raw,
                ks_int=self.ks_int,
            )

    def __sub__(self, other: Union[Self, jnp.ndarray]) -> FourierFieldSlice:
        return self + other * (-1.0) # type: ignore

    def __mul__(self, other: Union[Self, jnp.ndarray, float]) -> FourierFieldSlice:
        if isinstance(other, Field):
            if self.activate_jit_:
                new_name = ""
            else:
                try:
                    new_name = self.name + " * " + other.name
                except Exception:
                    new_name = "field"
            return FourierFieldSlice(
                self.fourier_domain,
                self.non_periodic_direction,
                self.data * other.data,
                new_name,
                *self.ks_raw,
                ks_int=self.ks_int,
            )
        else:
            if self.activate_jit_:
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
                self.fourier_domain,
                self.non_periodic_direction,
                self.data * other,
                new_name,
                *self.ks_raw,
                ks_int=self.ks_int,
            )

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other: float) -> FourierFieldSlice:
        if isinstance(other, Field):
            raise Exception("Don't know how to divide by another field")
        else:
            if self.activate_jit_:
                new_name = ""
            else:
                try:
                    if other.real >= 0:
                        new_name = self.name + "/" + str(other)
                    elif other == 1:
                        new_name = self.name
                    elif other == -1:
                        new_name = "-" + self.name
                    else:
                        new_name = self.name + "/ (" + str(other) + ") "
                except Exception:
                    new_name = "field"
            return FourierFieldSlice(
                self.fourier_domain,
                self.non_periodic_direction,
                self.data / other,
                new_name,
                *self.ks_raw,
                ks_int=self.ks_int,
            )

    def shift(self, value: float) -> FourierFieldSlice:
        out_field = self.data + value
        return FourierFieldSlice(self.fourier_domain, self.non_periodic_direction, out_field, name=self.name)

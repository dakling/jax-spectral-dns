#!/usr/bin/env python3

from __future__ import annotations

import time

from abc import ABC, abstractmethod
import math
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.figure as figure
from matplotlib.axes import Axes
import pyvista as pv
import h5py  # type: ignore

try:
    from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
except Exception:
    print("unable to load Axes3D, some plotting features may not work.")
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
except Exception:
    print("unable to load make_axes_locatable, some plotting features may not work.")
from matplotlib import colors
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import functools
import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import Self

if TYPE_CHECKING:
    from jax_spectral_dns._typing import AnyScalarField, AnyVectorField
    from jax_spectral_dns._typing import (
        jsd_array,
        np_float_array,
        np_complex_array,
        jsd_float,
        jnp_array,
        Vel_fn_type,
        np_jnp_array,
    )


import numpy as np

from pathlib import Path
import os
from shutil import copyfile

from jax_spectral_dns.domain import Domain, PhysicalDomain, FourierDomain, use_rfftn

NoneType = type(None)

T = TypeVar("T", bound="Field")


class Field(ABC):
    """Class that holds the information needed to describe a dependent variable
    and to perform operations on it."""

    plotting_dir = os.environ.get("JAX_SPECTRAL_DNS_PLOT_DIR", "./plots/")
    plotting_format = os.environ.get("JAX_SPECTRAL_DNS_PLOT_FORMAT", ".png")

    # setting this to True activates increases performance dramatically,
    # though some convenience features such as printing or plotting intermediate
    # state are disabled.
    activate_jit_ = False

    field_dir = os.environ.get("JAX_SPECTRAL_DNS_FIELD_DIR", "./fields/")

    def __init__(self, domain: Domain, data: "jnp_array", name: str):
        self.domain = domain
        # self.data = cast("jnp_array", jax.device_put(data, sharding))
        self.data = data
        self.name = name
        raise NotImplementedError("Trying to initialize an abstract class")

    @classmethod
    def Zeros(cls, domain: PhysicalDomain, name: str = "field") -> Self:
        data: "jnp_array" = jnp.zeros(domain.get_shape_aliasing())
        return cls(domain, data, name)

    @abstractmethod
    def get_domain(self) -> Domain: ...

    @abstractmethod
    def get_physical_domain(self) -> PhysicalDomain: ...

    @abstractmethod
    def normalize_by_max_value(self) -> Self:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        ...

    @abstractmethod
    def update_boundary_conditions(self) -> None:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        ...

    @classmethod
    def initialize(cls, cleanup: bool = True) -> None:
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
        newpaths = [Field.field_dir, Field.plotting_dir]
        for newpath in newpaths:
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        if cleanup:
            # clean plotting dir
            for file_ending in ["*.pdf", "*.png", "*.mp4", "*.txt"]:
                for f in Path(Field.plotting_dir).glob(file_ending):
                    if f.is_file():
                        f.unlink()

    def save_to_pickle_file(self, filename: str) -> None:
        """Save field to file filename using pickle."""
        field_array: "np_float_array" = np.array(self.data.tolist())
        field_array.dump(filename)

    def save_to_hdf_file(self, filename: str) -> None:
        """Save field to file filename using hdf5."""
        with h5py.File(filename, "w") as f:
            grp = f.create_group(str(self.get_time_step()))
            dset = grp.create_dataset(
                self.name, data=self.get_data(), compression="gzip", compression_opts=9
            )
            dset.attrs.create("time_step", self.time_step)

    def save_to_file(self, filename: str) -> None:
        """Save field to file filename."""
        filename = (
            filename
            if filename[0] in "./"
            else PhysicalField.field_dir + "/" + filename
        )
        try:
            self.save_to_hdf_file(filename)
        except Exception as e:
            print("unable to save as hdf due to the following exception:")
            print(e)
            self.save_to_pickle_file(filename)

    def get_data(self) -> jnp.ndarray:
        return self.data

    def get_name(self) -> str:
        """Return the name of the field."""
        return self.name

    def set_name(self, name: str) -> None:
        """Set the name of the field."""
        self.name = name

    def get_time_step(self) -> int:
        """Return the current time step of the field."""
        return self.time_step

    def set_time_step(self, time_step: int) -> None:
        """Set the current time step of the field."""
        self.time_step = time_step

    # @classmethod
    # def activate_jit(cls):
    #     cls.activate_jit_ = True

    @abstractmethod
    def __add__(self: T, _: Union[T, jnp.ndarray]) -> T: ...

    @abstractmethod
    def __sub__(self: T, _: Union[T, jnp.ndarray]) -> T: ...

    @abstractmethod
    def __mul__(self: T, _: Union[T, jnp.ndarray, "jsd_float"]) -> T: ...

    __rmul__ = __mul__
    __lmul__ = __mul__

    @abstractmethod
    def __truediv__(self: T, _: "jsd_float") -> T: ...

    @abstractmethod
    def shift(self: T, value: "jsd_float") -> T: ...

    def volume_integral(self) -> "jsd_float":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self.data)

    def __getitem__(self, index: Any) -> "jnp_array":
        return self.data[index]

    def max(self) -> "float":
        return cast(float, jnp.max(self.data.flatten()))

    def min(self) -> "float":
        return cast(float, jnp.min(self.data.flatten()))

    def absmax(self) -> "jnp_array":
        max = jnp.max(self.data.flatten())
        min = jnp.min(self.data.flatten())
        return jnp.max(jnp.array([abs(max), abs(min)]))
        # return max(abs(self.get_data().flatten()))

    def __neg__(self) -> Field:
        ret = self * (-1.0)
        ret.time_step = self.time_step
        return ret

    def __abs__(self) -> float:
        return cast(float, jnp.linalg.norm(self.data) / self.number_of_dofs())

    def number_of_dimensions(self) -> int:
        return len(self.all_dimensions())

    def number_of_dofs(self) -> int:
        out: int = 1
        for i in self.all_dimensions():
            out *= self.get_domain().number_of_cells(i)
        return out

    def all_dimensions(self) -> Sequence[int]:
        return self.get_domain().all_dimensions()

    def is_periodic(self, direction: int) -> bool:
        return self.get_domain().is_periodic(direction)

    def all_periodic_dimensions(self) -> List[int]:
        return self.get_domain().all_periodic_dimensions()

    def all_nonperiodic_dimensions(self) -> List[int]:
        return self.get_domain().all_nonperiodic_dimensions()

    def get_cheb_mat_2_homogeneous_dirichlet(self, direction: int) -> "np_float_array":
        return self.get_physical_domain().get_cheb_mat_2_homogeneous_dirichlet(
            direction
        )

    @abstractmethod
    def diff(self: T, direction: int, order: int = 1) -> T: ...

    def nabla(self: T) -> VectorField[T]:
        out = [self.diff(0)]
        out[0].name = "nabla_" + self.name + "_" + str(0)
        for dim in self.all_dimensions()[1:]:
            out.append(self.diff(dim))
            out[dim].name = "nabla_" + self.name + "_" + str(dim)
            out[dim].time_step = self.time_step
        return VectorField(out)

    def laplacian(self: T) -> T:
        out = self.diff(0, 2)
        for dim in self.all_dimensions()[1:]:
            out += self.diff(dim, 2)
        out.name = "lap_" + self.name
        out.time_step = self.time_step
        return out

    def plot_3d(
        self,
        direction: Optional[int] = None,
        coord: Optional[float] = None,
        rotate: bool = False,
        **params: Any,
    ) -> None:
        raise NotImplementedError()


class VectorField(Generic[T]):
    def __init__(self, elements: Sequence[T], name: Optional[str] = None):
        self.elements = elements
        self.name = name
        if name is not None:
            self.set_name(name)
        self.domain = elements[0].get_domain()

    @classmethod
    def Zeros(
        cls,
        field_cls: type["AnyScalarField"],
        domain: PhysicalDomain,
        name: str = "field",
    ) -> Self:
        dim = domain.number_of_dimensions
        fs = cast(List[T], [field_cls.Zeros(domain) for _ in range(dim)])
        return cls(fs, name)

    @classmethod
    def FromRandom(
        cls,
        field_cls: type["AnyScalarField"],
        domain: PhysicalDomain,
        seed: "jsd_float" = 0,
        energy_norm: float = 1.0,
        name: str = "field",
    ) -> Self:
        """Construct a random field depending on the independent variables described by domain."""
        dim = domain.number_of_dimensions
        fs = cast(
            List[T],
            [field_cls.FromRandom(domain, seed, energy_norm) for _ in range(dim)],
        )
        return cls(fs, name)

    @classmethod
    def FromData(
        cls,
        field_cls: type["AnyScalarField"],
        domain: PhysicalDomain,
        data: "jnp_array",
        name: str = "field",
        allow_projection: bool = False,
        dim: Optional[int] = None,
    ) -> Self:
        if dim is None:
            dim = domain.number_of_dimensions
        if field_cls is PhysicalField:
            data_matches_domain = data.shape[1:] == domain.get_shape_aliasing()
        elif field_cls is FourierField:
            data_matches_domain = data.shape[1:] == domain.get_shape()
        else:
            raise Exception(field_cls, "not known.")

        if not allow_projection:
            assert data_matches_domain, (
                "Data in provided file (shape: "
                + str(data.shape[1:])
                + ") does not match domain (shape: "
                + str(domain.get_shape_aliasing())
                + " ). Call with allow_projection=True if you would like to automatically project the data onto the provided domain."
            )
        if not data_matches_domain:
            data_domain_shape = tuple(
                [
                    (
                        int(data.shape[i + 1] / domain.aliasing)
                        if domain.is_periodic(i)
                        else data.shape[i + 1]
                    )
                    for i in domain.all_dimensions()
                ]
            )
            data_domain = PhysicalDomain.create(
                data_domain_shape,
                domain.periodic_directions,
                domain.scale_factors,
                domain.aliasing,
                domain.dealias_nonperiodic,
            )
            if field_cls is PhysicalField:
                fs = cast(
                    List[T],
                    [
                        cast(PhysicalField, field_cls(data_domain, data[i]))
                        .hat()
                        .project_onto_domain(domain)
                        .no_hat()
                        for i in range(dim)
                    ],
                )
            elif field_cls is FourierField:
                fs = cast(
                    List[T],
                    [
                        cast(
                            FourierField, field_cls(data_domain, data[i])
                        ).project_onto_domain(domain)
                        for i in range(dim)
                    ],
                )
            else:
                raise Exception(field_cls, "not known.")
        else:
            fs = cast(List[T], [field_cls(domain, data[i]) for i in range(dim)])
        return cls(fs, name)

    @classmethod
    def read_pickle(cls, filename: str, _: str) -> "jnp_array":
        field_array = np.load(filename, allow_pickle=True)
        data = jnp.array(field_array.tolist())
        return data

    @classmethod
    def read_hdf(
        cls, filename: str, name: str, time_step: int
    ) -> Tuple["jnp_array", int]:
        with h5py.File(filename, "r") as f:
            if time_step < 0:
                time_step = int([name for name in f][-1]) + time_step + 1
            grp = f[str(time_step)]
            assert grp is not None, (
                "time step "
                + str(time_step)
                + " not found, only "
                + str([name for name in f])
                + " found."
            )
            dset = grp.get(name)
            assert dset is not None, (
                "dataset "
                + name
                + " not found, only "
                + str([name for name in grp])
                + " found."
            )
            return (jnp.array(dset), time_step)

    @classmethod
    def FromFile(
        cls,
        domain: PhysicalDomain,
        filename: str,
        name: str = "field",
        time_step: int = -1,
        allow_projection: bool = False,
    ) -> VectorField[PhysicalField]:
        """Construct new field depending on the independent variables described
        by domain by reading in a saved field from file filename."""
        filename = (
            filename
            if filename[0] in "./"
            else PhysicalField.field_dir + "/" + filename
        )
        try:
            field_array, time_step = cls.read_hdf(filename, name, time_step)
        except Exception as e:
            # print("unable to load hdf due to the following exception:")
            # print(e)
            # print("trying to interpret file as pickle instead")
            field_array = cls.read_pickle(filename, name)
        out: VectorField[PhysicalField] = VectorField.FromData(
            PhysicalField, domain, field_array, name, allow_projection
        )
        out.set_time_step(max(0, time_step))
        return out

    def project_onto_domain(
        self: VectorField[FourierField], domain: PhysicalDomain
    ) -> VectorField[FourierField]:
        return VectorField(
            [f.project_onto_domain(domain) for f in self], name=self.name
        )

    def filter(
        self: VectorField[FourierField], domain: PhysicalDomain
    ) -> VectorField[FourierField]:
        return VectorField([f.filter() for f in self], name=self.name)

    def __getitem__(self, index: int) -> T:
        return self.elements[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __str__(self) -> str:
        out = ""
        for elem in self.elements:
            out += str(elem)
        return out

    def __add__(self, other: Union[VectorField[T], "jnp_array"]) -> VectorField[T]:
        out = []
        for i in range(len(self)):
            f: T = self[i]
            other_i: Union[T, "jnp_array"] = other[i]
            out_: T = f + other_i
            out.append(out_)
        return VectorField(out)

    def __sub__(self, other: Union[VectorField[T], "jnp_array"]) -> VectorField[T]:
        return self + (-1) * other

    def __mul__(
        self, other: Union[VectorField[T], "jnp_array", "jsd_float"]
    ) -> VectorField[T]:
        out = []
        if isinstance(other, VectorField):
            for i in range(len(self)):
                f: T = self[i]
                out_: T = f * other[i]
                out.append(out_)
        else:
            for i in range(len(self)):
                f = self[i]
                out_ = f * other
                out.append(out_)
        return VectorField(out)

    __rmul__ = __mul__
    __lmul__ = __mul__

    def __truediv__(self, other: "jsd_float") -> VectorField[T]:
        out = []
        for i in range(len(self)):
            out.append(self[i] / other)
        return VectorField(out)

    def __pow__(
        self: VectorField[PhysicalField], exponent: float
    ) -> VectorField[PhysicalField]:
        out = []
        for i in range(len(self)):
            out.append(self[i] ** exponent)
        return VectorField(out)

    def shift(self, value: "jnp_array") -> VectorField[T]:
        out = []
        assert len(value) == len(self), "Dimension mismatch."
        for i in range(len(self)):
            out.append(self[i].shift(value[i]))
        return VectorField(out)

    def max(self) -> float:
        return max([f.max() for f in self])

    def min(self) -> float:
        return min([f.min() for f in self])

    def laplacian(self: VectorField[T]) -> VectorField[T]:
        out = []
        for f in self:
            out_ = f.diff(0, 2)
            for dim in self.all_dimensions()[1:]:
                out_ += f.diff(dim, 2)
            out.append(out_)
        out_field = VectorField(out)
        out_field.name = "lap_" + self.get_name()
        out_field.time_step = self.get_time_step()
        return out_field

    def get_physical_domain(self) -> PhysicalDomain:
        f = self[0]
        if type(f) is PhysicalField:
            return f.get_domain()
        elif type(f) is FourierField:
            return f.get_physical_domain()
        else:
            raise Exception("Unable to return PhysicalDomain.")

    def get_fourier_domain(self) -> FourierDomain:
        f = self[0]
        if type(f) is PhysicalField:
            raise Exception("PhysicalField has no attribute FourierDomain.")
        elif type(f) is FourierField:
            return f.get_domain()
        else:
            raise Exception("Unable to return FourierDomain.")

    def __abs__(self) -> float:
        out: float = 0.0
        for f in self:
            out += abs(f)
        return out

    def magnitude(self: VectorField[PhysicalField]) -> PhysicalField:
        out = self[0] ** 2
        for i in range(1, len(self)):
            out += self[i] ** 2
        out.set_name(self.get_name() + "_magnitude")
        out.set_time_step(self.time_step)
        return out

    def definite_integral(
        self: VectorField[PhysicalField], direction: int
    ) -> VectorField[PhysicalField]:
        return VectorField(
            [
                cast(PhysicalField, self[i].definite_integral(direction))
                for i in range(len(self))
            ]
        )

    def energy(self: VectorField[PhysicalField]) -> float:
        en: float = 0.0
        for f in self:
            en += f.energy()
        return en

    def energy_p(self: VectorField[PhysicalField], p: float = 1.0) -> float:
        energy_field = PhysicalField.Zeros(self.get_physical_domain())
        for f in self:
            energy_field += 0.5 * f * f

        energy_field_p = energy_field**p
        domain_volume: float = cast(
            float,
            2.0 ** (len(self[0].all_nonperiodic_dimensions()))
            * jnp.prod(jnp.array(self.get_physical_domain().scale_factors)),
        )  # nonperiodic dimensions are size 2, but its scale factor is only 1
        return cast(
            float, ((energy_field_p.volume_integral()) / domain_volume) ** (1.0 / p)
        )

    def field_2d(
        self: VectorField[FourierField], direction: int, wavenumber: int = 0
    ) -> VectorField[FourierField]:
        return VectorField([f.field_2d(direction, wavenumber) for f in self])

    def energy_2d(self: VectorField[FourierField], direction: int) -> float:
        en: float = 0.0
        for f in self:
            en += f.energy_2d(direction)
        return en

    def get_localisation(self: VectorField[PhysicalField], p: int = 3) -> float:
        return self.energy_p(p) / self.energy()

    def get_q_criterion(self: VectorField[PhysicalField]) -> PhysicalField:
        grad_u = [
            [self[i].diff(j) for j in range(self[0].number_of_dimensions())]
            for i in range(self[0].number_of_dimensions())
        ]
        u_grad_sq = [
            [
                (grad_u[i][j] * grad_u[j][i])
                for j in range(self[0].number_of_dimensions())
            ]
            for i in range(self[0].number_of_dimensions())
        ]
        u_grad_sq_flat = [x for xs in u_grad_sq for x in xs]
        u_grad_sum = functools.reduce(lambda a, b: a + b, u_grad_sq_flat)
        return -0.5 * u_grad_sum

    def normalize_by_energy(
        self: VectorField[PhysicalField],
    ) -> VectorField[PhysicalField]:
        """Divide each field by the energy of the Vector field."""
        en = self.energy()
        for f in self:
            f.data = f.data / jnp.sqrt(en)
        return self

    def normalize_by_max_value(self) -> Self:
        """Divide each field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        for f in self:
            f = f.normalize_by_max_value()
        return self

    def normalize_by_flow_rate(
        self: VectorField[PhysicalField], flow_direction: int, direction: int
    ) -> VectorField[PhysicalField]:
        flow_rate = self[flow_direction].get_flow_rate(direction)
        for f in self:
            f.data = jax.lax.cond(
                flow_rate > 1e-20, lambda: f.data / flow_rate, lambda: f.data
            )
        return self

    def energy_norm(self: VectorField[PhysicalField], k: float) -> float:
        energy = k**2 * self[1] * self[1]
        energy += self[1].diff(1) * self[1].diff(1)
        vort = self.curl()
        energy += vort[1] * vort[1]
        return energy.volume_integral()

    def save_to_pickle_file(self, filename: str) -> None:
        """Save field to file filename using pickle.

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
        field_array = np.array(self.get_data().tolist())
        field_array.dump(filename)

    def save_to_hdf_file(self, filename: str) -> None:
        """Save field to file filename using hdf5.

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
        with h5py.File(filename, "w") as f:
            grp = f.create_group(str(self.get_time_step()))
            dset = grp.create_dataset(
                self.name, data=self.get_data(), compression="gzip", compression_opts=9
            )

    def save_to_file(self, filename: str) -> None:
        """Save field to file filename.

        Note: the resulting format is compatible with dedalus. Importing e.g. a
        velocity field into dedalus is as straightforward as:
        u_array = np.load("/path/to/u_file", allow_pickle=True)
        v_array = np.load("/path/to/v_file", allow_pickle=True)
        w_array = np.load("/path/to/w_file", allow_pickle=True)

        ... # dedalus case setup goes here...
        u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
        u.data = np.stack([u_array, v_array, w_array])"""
        filename = (
            filename
            if filename[0] in "./"
            else PhysicalField.field_dir + "/" + filename
        )
        try:
            self.save_to_hdf_file(filename)
        except Exception as e:
            print("unable to save as hdf due to the following exception:")
            print(e)
            self.save_to_pickle_file(filename)

    def get_data(self) -> "jnp_array":
        return jnp.array([f.data for f in self])
        # return [f.data for f in self]

    def get_time_step(self) -> int:
        time_steps = [f.time_step for f in self]
        out = max(time_steps)
        assert type(out) is int
        return out

    def set_time_step(self, time_step: int) -> None:
        self.time_step = time_step
        for j in range(len(self)):
            self[j].set_time_step(time_step)

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

    def update_boundary_conditions(self) -> None:
        for field in self:
            field.update_boundary_conditions()

    def all_dimensions(self) -> Sequence[int]:
        out = self[0].get_domain().all_dimensions()
        # assert type(out) is Sequence[int]
        return out

    def all_periodic_dimensions(self) -> List[int]:
        out = self[0].get_domain().all_periodic_dimensions()
        # assert type(out) is list[int]
        return out

    def all_nonperiodic_dimensions(self) -> List[int]:
        out = self[0].get_domain().all_nonperiodic_dimensions()
        # assert type(out) is list[int]
        return out

    def hat(self: VectorField[PhysicalField]) -> VectorField[FourierField]:
        out = VectorField([f.hat() for f in self])
        out.set_name(out.get_name())
        out.set_time_step(self.get_time_step())
        return out

    def no_hat(self: VectorField[FourierField]) -> VectorField[PhysicalField]:
        out = VectorField([f.no_hat() for f in self])
        out.set_name(out.get_name())
        out.set_time_step(self.get_time_step())
        return out

    def plot(self, *other_fields: VectorField[PhysicalField]) -> None:
        try:
            for i in jnp.arange(len(self)):
                other_fields_i = [item[i] for item in other_fields]
                f = self[i]
                assert (
                    type(f) is PhysicalField
                ), "plot only implemented for PhysicalField."
                f.plot(*other_fields_i)
        except Exception as e:
            print(
                "VectorField[PhysicalField].plot failed with the following exception:"
            )
            print(e)
            print("ignoring this and carrying on.")

    def plot_3d(
        self,
        direction: Optional[int] = None,
        coord: Optional[float] = None,
        rotate: bool = False,
        **params: Any,
    ) -> None:
        try:
            for f in self:
                f.plot_3d(direction, coord, rotate, **params)
        except Exception as e:
            print("VectorField.plot_3d failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_wavenumbers(self, direction: int) -> None:
        try:
            for f in self:
                assert (
                    type(f) is PhysicalField
                ), "plot_wavenumbers only implemented for PhysicalField."
                f.plot_wavenumbers(direction)
        except Exception as e:
            print("VectorField.plot_wavenumbers failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_isosurfaces(self, iso_val: float = 0.4) -> None:
        try:
            for f in self:
                assert (
                    type(f) is PhysicalField
                ), "plot_isosurfaces only implemented for PhysicalField."
                f.plot_isosurfaces(iso_val)
        except Exception as e:
            print(
                "VectorField[PhysicalField].plot_isosurfaces failed with the following exception:"
            )
            print(e)
            print("ignoring this and carrying on.")

    def plot_q_criterion_isosurfaces(
        self: VectorField[PhysicalField],
        plot_min_and_max: bool = False,
        iso_vals: List[float] = [0.05, 0.1, 0.5],
    ) -> None:
        q_crit = self.get_q_criterion()
        q_crit.set_time_step(self.get_time_step())
        q_crit.set_name("q_criterion")
        q_crit.plot_isosurfaces(
            iso_vals[0], plot_min_and_max=plot_min_and_max, other_vals=iso_vals[1:]
        )

    def cross_product(self, other: VectorField[T]) -> VectorField[T]:
        out_0: T = self[1] * other[2] - self[2] * other[1]
        out_1: T = self[2] * other[0] - self[0] * other[2]
        out_2: T = self[0] * other[1] - self[1] * other[0]

        time_step = self.get_time_step()
        out = [out_0, out_1, out_2]
        for f in out:
            f.time_step = time_step

        return VectorField(out)

    def curl(self) -> VectorField[T]:
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

        curl_0: T = w_y - v_z
        curl_1: T = u_z - w_x
        curl_2: T = v_x - u_y

        time_step = self.get_time_step()
        out = [curl_0, curl_1, curl_2]
        for f in out:
            f.time_step = time_step

        return VectorField(out)

    def div(self) -> T:
        out = self[0].diff(0)
        for dim in self.all_dimensions()[1:]:
            out = out + self[dim].diff(dim)
        if self.name is not None:
            out.name = "div_" + self.name
        else:
            out.name = "div_field"
        return out

    def reconstruct_from_wavenumbers(
        self, fn: Callable[[int, int], "jnp_array"], number_of_other_fields: int = 0
    ) -> Tuple[VectorField[FourierField], int]:

        # jit = True
        # vectorize = True

        jit = False
        # vectorize = jax.devices()[0].platform == "gpu"  # True on GPUs and False on CPUs
        vectorize = False

        physical_domain = self.get_physical_domain()
        fourier_domain = self.get_fourier_domain()

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
                fourier_domain.grid[self.all_periodic_dimensions()[0]].astype(int)
            )
            k2s = jnp.array(
                fourier_domain.grid[self.all_periodic_dimensions()[1]].astype(int)
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
                        physical_domain,
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
                            physical_domain,
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
                        physical_domain,
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

    def plot_streamlines(self, normal_direction: int) -> None:
        try:
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
            del fig, ax
        except Exception as e:
            print("plot_streamlines failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_vectors(self, normal_direction: int) -> None:
        try:
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
            del fig, ax
        except Exception as e:
            print("plot_vectors failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")


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
        data: "jnp_array",
        name: str = "field",
        time_step: int = 0,
    ):
        self.physical_domain = domain
        self.data = data
        self.name = name
        self.time_step: int = time_step

    def shift(self, value: "jsd_float") -> PhysicalField:
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
                if len(other.name) > 0 and other.name[0] == "-":
                    new_name = self.name + " - " + other.name[1:]
                else:
                    new_name = self.name + " + " + other.name
            ret = PhysicalField(
                self.physical_domain, self.data + other.data, name=new_name
            )
        else:
            assert isinstance(other, jnp.ndarray)
            ret = PhysicalField(self.physical_domain, self.data + other, name="field")
        ret.time_step = self.time_step
        return ret

    def __sub__(self, other: Union[Self, jnp.ndarray]) -> PhysicalField:
        assert not isinstance(
            other, FourierField
        ), "Attempted to subtract a Field and a Fourier Field."
        return self + other * (-1.0)  # type: ignore

    def __mul__(self, other: Union[Self, jnp.ndarray, "jsd_float"]) -> PhysicalField:
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

    def __truediv__(self, other: "jsd_float") -> PhysicalField:
        if isinstance(other, Field):
            raise Exception("Don't know how to divide by another field")
        else:
            new_name = "field"
            ret = PhysicalField(self.physical_domain, self.data / other, name=new_name)
            ret.time_step = self.time_step
            return ret

    def __pow__(self, exponent: float) -> PhysicalField:
        out_data = self.data**exponent
        return PhysicalField(
            self.get_physical_domain(), out_data, name=self.name + "^" + str(exponent)
        )

    @classmethod
    def FromFunc(
        cls,
        domain: PhysicalDomain,
        func: Optional["Vel_fn_type"] = None,
        name: str = "field",
    ) -> Self:
        """Construct from function func depending on the independent variables described by domain."""
        if not func:
            func_: "Vel_fn_type" = lambda x: 0.0 * math.prod(x)
        else:
            assert func is not None
            func_ = func
        field = jnp.array(list(map(lambda *x: func_(x), *domain.mgrid)))
        return cls(domain, field, name)

    @classmethod
    def FromRandom(
        cls,
        domain: PhysicalDomain,
        seed: "jsd_float" = 0,
        energy_norm: float = 1.0,
        name: str = "field",
    ) -> PhysicalField:
        """Construct a random field depending on the independent variables described by domain."""
        # TODO generate "nice" random fields
        interval = (-1.0, 1.0)
        key = jax.random.PRNGKey(seed)
        zero_field = PhysicalField.FromFunc(domain)
        rands = []
        for _ in jnp.arange(zero_field.number_of_dofs()):
            key, subkey = jax.random.split(key)
            rands.append(
                jax.random.uniform(subkey, minval=interval[0], maxval=interval[1])
            )
        field = jnp.array(rands).reshape(zero_field.get_domain().get_shape_aliasing())
        out = cls(domain, field, name)
        # smooth_field = PhysicalField.FromFunc(
        #     domain, lambda X: jnp.exp(-((1.4 * X[1]) ** 8))
        # )  # make sure that we are not messing with the boundary conditions
        # out *= smooth_field
        out.update_boundary_conditions()
        out.normalize_by_energy()
        out *= jnp.sqrt(energy_norm)
        out.set_name(name)
        return out

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
    def read_pickle(cls, filename: str, _: str) -> "jnp_array":
        field_array = np.load(filename, allow_pickle=True)
        data = jnp.array(field_array.tolist())
        return data

    @classmethod
    def read_hdf(cls, filename: str, name: str, time_step: int) -> "jnp_array":
        with h5py.File(filename, "r") as f:
            if time_step < 0:
                time_step = int([name for name in f][-1]) + time_step + 1
            grp = f[str(time_step)]
            assert grp is not None, (
                "time step "
                + str(time_step)
                + " not found, only "
                + str([name for name in f])
                + " found."
            )
            dset = grp.get(name)
            assert dset is not None, (
                "dataset "
                + name
                + " not found, only "
                + str([name for name in grp])
                + " found."
            )
            return jnp.array(dset)

    @classmethod
    def FromFile(
        cls,
        domain: PhysicalDomain,
        filename: str,
        name: str = "field",
        time_step: int = -1,
        allow_projection: bool = False,
    ) -> PhysicalField:
        """Construct new field depending on the independent variables described
        by domain by reading in a saved field from file filename."""
        filename = (
            filename
            if filename[0] in "./"
            else PhysicalField.field_dir + "/" + filename
        )
        try:
            data = cls.read_hdf(filename, name, time_step)
        except Exception as e:
            # print("unable to load hdf due to the following exception:")
            # print(e)
            # print("trying to interpret file as pickle instead")
            data = cls.read_pickle(filename, name)
        data_matches_domain = data.shape == domain.get_shape_aliasing()
        if not allow_projection:
            assert (
                data_matches_domain
            ), "Data in provided file does not match domain. Call with allow_projection=True if you would like to automatically project the data onto the provided domain."
        if not data_matches_domain:
            data_domain_shape = tuple(
                [
                    (
                        int(data.shape[i] / domain.aliasing)
                        if domain.is_periodic(i)
                        else data.shape[i]
                    )
                    for i in domain.all_dimensions()
                ]
            )
            data_domain = PhysicalDomain.create(
                data_domain_shape,
                domain.periodic_directions,
                domain.scale_factors,
                domain.aliasing,
                domain.dealias_nonperiodic,
            )
            out = (
                PhysicalField(data_domain, data, name=name)
                .hat()
                .project_onto_domain(domain)
                .no_hat()
            )
        else:
            out = PhysicalField(domain, data, name=name)
        return out

    def normalize_by_max_value(self) -> Self:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        max: float = cast(float, abs(self.absmax()))

        self.data = jax.lax.cond(
            max > 1e-20, lambda: self.data / max, lambda: self.data
        )
        return self

    def get_domain(self) -> PhysicalDomain:
        return self.physical_domain

    def get_physical_domain(self) -> PhysicalDomain:
        return self.get_domain()

    def l2error(self, fn: "Vel_fn_type") -> "jsd_array":
        # TODO supersampling
        analytical_solution = PhysicalField.FromFunc(self.physical_domain, fn)
        return cast(
            "jsd_array", jnp.linalg.norm((self - analytical_solution).data, None)
        )

    def volume_integral(self) -> float:
        int = PhysicalField(self.physical_domain, self.data)
        dims = list(reversed(self.all_dimensions()))
        for i in dims[:-1]:
            assert type(int) is PhysicalField, type(int)
            out_ = int.definite_integral(i)
            assert type(out_) is PhysicalField, type(out_)
            int = out_
        assert type(int) is PhysicalField, type(int)
        out = int.definite_integral(dims[-1])
        return cast(float, out)

    def energy(self) -> float:
        energy = 0.5 * self * self
        domain_volume = 2.0 ** (len(self.all_nonperiodic_dimensions())) * jnp.prod(
            jnp.array(self.physical_domain.scale_factors)
        )  # nonperiodic dimensions are size 2, but its scale factor is only 1
        return cast(float, energy.volume_integral() / domain_volume)

    def energy_p(self, p: float = 1.0) -> float:
        energy_p = (0.5 * self * self) ** p
        domain_volume = 2.0 ** (len(self.all_nonperiodic_dimensions())) * jnp.prod(
            jnp.array(self.physical_domain.scale_factors)
        )  # nonperiodic dimensions are size 2, but its scale factor is only 1
        return cast(float, ((energy_p.volume_integral()) / domain_volume) ** (1 / p))

    def inf_norm(self) -> float:
        return cast(float, self.absmax())

    def get_localisation(self: PhysicalField, p: int = 3) -> float:
        return self.energy_p(p) / self.energy()

    def normalize_by_energy(self) -> Self:
        en = self.energy()
        self.data = jax.lax.cond(en > 1e-20, lambda: self.data / en, lambda: self.data)
        return self

    def normalize_by_energy_p(self, p: float = 1.0) -> Self:
        en = self.energy_p(p)
        self.data = jax.lax.cond(en > 1e-20, lambda: self.data / en, lambda: self.data)
        return self

    def normalize_by_flow_rate(self, direction: int) -> Self:
        flow_rate = self.get_flow_rate(direction)
        self.data = jax.lax.cond(
            flow_rate > 1e-20, lambda: self.data / flow_rate, lambda: self.data
        )
        return self

    def get_flow_rate(self, direction: int) -> "jsd_float":
        # TODO this assumes a 3D field that is constant in z
        int: PhysicalField = self.definite_integral(direction)  # type: ignore[assignment]
        return cast("jsd_float", int[0, 0])

    def update_boundary_conditions(self) -> None:
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

    def eval(self, X: Sequence[float]) -> "jsd_array":
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

    def plot_center(
        self, dimension: int, *other_fields: PhysicalField, **params: Any
    ) -> None:
        try:
            ax = params.get("ax")
            if ax is None:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
            else:
                fig = cast("figure.Figure", params.get("fig"))
            name = params.get("name", self.name)
            other_names = params.get("other_names", [f.name for f in other_fields])
            rotate = params.get("rotate", False)
            if self.physical_domain.number_of_dimensions == 1:
                assert type(ax) is Axes
                if not rotate:
                    ax.plot(self.physical_domain.grid[0], self.data, label=name)
                else:
                    ax.plot(self.data, self.physical_domain.grid[0], label=name)
                i = 0
                for other_field in other_fields:
                    if not rotate:
                        ax.plot(
                            other_field.physical_domain.grid[dimension],
                            other_field.data,
                            "--",
                            label=other_names[i],
                        )
                    else:
                        ax.plot(
                            other_field.data,
                            other_field.physical_domain.grid[dimension],
                            "--",
                            label=other_names[i],
                        )
                    i += 1
                if params.get("ax") is None:
                    fig.legend()

                def save() -> None:
                    if params.get("ax") is None:
                        assert fig is not None
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
                assert type(ax) is Axes
                other_dim = [i for i in self.all_dimensions() if i != dimension][0]
                N_c = self.physical_domain.number_of_cells(other_dim) // 2
                if not rotate:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_c, axis=other_dim),
                        label=name,
                    )
                else:
                    ax.plot(
                        self.data.take(indices=N_c, axis=other_dim),
                        self.physical_domain.grid[dimension],
                        label=name,
                    )
                i = 0
                for other_field in other_fields:
                    if not rotate:
                        ax.plot(
                            other_field.physical_domain.grid[dimension],
                            other_field.data.take(indices=N_c, axis=other_dim),
                            "--",
                            label=other_names[i],
                        )
                    else:
                        ax.plot(
                            other_field.data.take(indices=N_c, axis=other_dim),
                            other_field.physical_domain.grid[dimension],
                            "--",
                            label=other_names[i],
                        )
                    i += 1
                if params.get("ax") is None:
                    fig.legend()

                def save() -> None:
                    if params.get("ax") is None:
                        assert fig is not None
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
                assert type(ax) is Axes
                other_dims = [i for i in self.all_dimensions() if i != dimension]
                N_cs = [
                    self.physical_domain.number_of_cells(dim) // 2 for dim in other_dims
                ]
                if not rotate:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                            indices=N_cs[0], axis=other_dims[0]
                        ),
                        label=self.name,
                    )
                else:
                    ax.plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                            indices=N_cs[0],
                            axis=other_dims[0],
                        ),
                        label=self.name,
                    )
                i = 0
                for other_field in other_fields:
                    if not rotate:
                        ax.plot(
                            other_field.physical_domain.grid[dimension],
                            other_field.data.take(
                                indices=N_cs[1], axis=other_dims[1]
                            ).take(indices=N_cs[0], axis=other_dims[0]),
                            "--",
                            label=other_names[i],
                        )
                    else:
                        ax.plot(
                            other_field.physical_domain.grid[dimension],
                            other_field.data.take(
                                indices=N_cs[1], axis=other_dims[1]
                            ).take(
                                indices=N_cs[0],
                                axis=other_dims[0],
                            ),
                            "--",
                            label=other_names[i],
                        )
                    i += 1
                if params.get("ax") is None:
                    fig.legend()

                def save() -> None:
                    if params.get("ax") is None:
                        assert fig is not None
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
            del fig, ax
        except Exception as e:
            print("plot_center failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot(self, *other_fields: PhysicalField, **params: Any) -> None:
        try:
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
                        other_field.physical_domain.grid[0],
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
                del ax

            elif self.physical_domain.number_of_dimensions == 2:
                fig = figure.Figure(figsize=(15, 5))
                ax_ = fig.subplots(1, 1)
                # assert type(ax_) is np.ndarray
                assert type(ax_) is Axes
                ims = []
                data = self.data
                rotate = params.get("rotate", False)
                other_dim = list(self.all_dimensions())
                if rotate:
                    other_dim.reverse()
                    data = data.T

                extent = (
                    self.physical_domain.grid[other_dim[0]][0],
                    self.physical_domain.grid[other_dim[0]][-1],
                    self.physical_domain.grid[other_dim[1]][0],
                    self.physical_domain.grid[other_dim[1]][-1],
                )
                x = self.physical_domain.grid[other_dim[0]]
                y = jnp.flip(self.physical_domain.grid[other_dim[1]])
                Nx = self.physical_domain.get_shape()[other_dim[0]]
                Ny = self.physical_domain.get_shape()[other_dim[1]]
                xi = np.linspace(x[0], x[-1], Nx)
                yi = np.linspace(y[0], y[-1], Ny)
                interp = RegularGridInterpolator((x, y), data, method="cubic")
                interp_data = np.array(
                    [[interp([[x_, y_]])[0] for x_ in xi] for y_ in yi]
                )
                ims.append(
                    ax_.imshow(
                        interp_data,
                        interpolation=None,
                        extent=extent,
                    )
                )
                ax_.set_xlabel("$" + "xyz"[other_dim[0]] + "$")
                ax_.set_ylabel("$" + "xyz"[other_dim[1]] + "$")
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
                vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in ims:
                    im.set_norm(norm)
                    name = params.get("name", self.name)
                    name_color = params.get("name_color", "black")
                    divider = make_axes_locatable(ax_)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(
                        ims[0], cax=cax, label=name, orientation="vertical"
                    )
                    cbar.ax.yaxis.label.set_color(name_color)
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
                    del ax_
            elif self.physical_domain.number_of_dimensions == 3:
                fig = figure.Figure()
                # ax = fig.subplots(1, 3, figsize=(15, 5))
                ax__ = fig.subplots(1, 3)
                assert type(ax__) is np.ndarray
                for dimension in self.all_dimensions():
                    other_dims = [i for i in self.all_dimensions() if i != dimension]
                    N_cs = [
                        self.physical_domain.number_of_cells(dim) // 2
                        for dim in other_dims
                    ]
                    ax__[dimension].plot(
                        self.physical_domain.grid[dimension],
                        self.data.take(indices=N_cs[1], axis=other_dims[1]).take(
                            indices=N_cs[0], axis=other_dims[0]
                        ),
                        label=self.name,
                    )
                    for other_field in other_fields:
                        ax__[dimension].plot(
                            other_field.physical_domain.grid[dimension],
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
                del ax__
            else:
                raise Exception("Not implemented yet")
            del fig
        except Exception as e:
            raise e
            print("plot failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_3d(
        self,
        direction: Optional[int] = None,
        coord: Optional[float] = None,
        rotate: bool = False,
        **params: Any,
    ) -> None:
        try:
            if direction is not None:
                self.plot_3d_single(direction, coord, rotate, **params)
            else:
                assert (
                    self.physical_domain.number_of_dimensions == 3
                ), "Only 3D supported for this plotting method."
                try:
                    fig = figure.Figure(layout="constrained")
                except Exception:
                    fig = figure.Figure()
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
                    data_shape = self.data.shape
                    if coord is None:
                        N_c = (data_shape[dim] - 1) // 2
                        coord = (
                            N_c
                            / data_shape[dim]
                            * (
                                self.get_domain().grid[dim][-1]
                                - self.get_domain().grid[dim][0]
                            )
                        )
                    else:
                        N_c = int(
                            (data_shape[dim] - 1)
                            * (coord - self.get_domain().grid[dim][0])
                            / (
                                self.get_domain().grid[dim][-1]
                                - self.get_domain().grid[dim][0]
                            )
                        )
                    other_dim = [i for i in self.all_dimensions() if i != dim]

                    extent = (
                        self.physical_domain.grid[other_dim[0]][0],
                        self.physical_domain.grid[other_dim[0]][-1],
                        self.physical_domain.grid[other_dim[1]][0],
                        self.physical_domain.grid[other_dim[1]][-1],
                    )
                    x = self.physical_domain.grid[other_dim[0]]
                    y = jnp.flip(self.physical_domain.grid[other_dim[1]])
                    Nx = self.physical_domain.get_shape()[other_dim[0]]
                    Ny = self.physical_domain.get_shape()[other_dim[1]]
                    xi = np.linspace(x[0], x[-1], Nx)
                    yi = np.linspace(y[0], y[-1], Ny)
                    interp = RegularGridInterpolator((x, y), self.data, method="cubic")
                    interp_data = np.array(
                        [[interp([[x_, y_]])[0] for x_ in xi] for y_ in yi]
                    )
                    ims.append(
                        ax[dim].imshow(
                            interp_data.take(indices=N_c, axis=dim),
                            interpolation=None,
                            extent=extent,
                        )
                    )
                    # ims.append(
                    #     ax[dim].imshow(
                    #         self.data.take(indices=N_c, axis=dim),
                    #         interpolation=None,
                    #         extent=(
                    #             self.physical_domain.grid[other_dim[1]][0],
                    #             self.physical_domain.grid[other_dim[1]][-1],
                    #             self.physical_domain.grid[other_dim[0]][0],
                    #             self.physical_domain.grid[other_dim[0]][-1],
                    #         ),
                    #     )
                    # )
                    ax[dim].set_xlabel("$" + "xyz"[other_dim[1]] + "$")
                    ax[dim].set_ylabel("$" + "xyz"[other_dim[0]] + "$")
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
                vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in ims:
                    im.set_norm(norm)
                name = params.get("name", self.name)
                name_color = params.get("name_color", "black")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(ims[0], cax=cax, label=name)
                cbar.ax.yaxis.label.set_color(name_color)
                assert coord is not None
                ax[dim].set_title(
                    "$" + "xyz"[dim] + " = " + "{:.2f}".format(coord) + "$"
                )

                def save() -> None:
                    fig.savefig(
                        self.plotting_dir
                        + "plot_3d_"
                        + self.name
                        + "_latest"
                        + self.plotting_format,
                        bbox_inches="tight",
                    )
                    fig.savefig(
                        self.plotting_dir
                        + "plot_3d_"
                        + self.name
                        + "_t_"
                        + "{:06}".format(self.time_step)
                        + self.plotting_format,
                        bbox_inches="tight",
                    )

                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
                del fig, ax
        except Exception:
            for i in self.all_dimensions():
                try:
                    self.plot_3d_single(i, rotate, **params)
                except Exception as e:
                    print("plot_3d failed with the following exception:")
                    print(e)
                    print("ignoring this and carrying on.")

    def plot_3d_single(
        self,
        dim: int,
        coord: Optional[float] = None,
        rotate: bool = False,
        **params: Any,
    ) -> None:
        try:
            assert (
                self.physical_domain.number_of_dimensions == 3
            ), "Only 3D supported for this plotting method."
            ax = params.get("ax")
            if ax is None:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
            else:
                fig = cast("figure.Figure", params.get("fig"))
            assert type(ax) is Axes
            ims = []
            data_shape = self.data.shape
            if coord is None:
                N_c = (data_shape[dim] - 1) // 2
                coord = (
                    N_c
                    / data_shape[dim]
                    * (self.get_domain().grid[dim][-1] - self.get_domain().grid[dim][0])
                )
            else:
                N_c = int(
                    data_shape[dim]
                    * (coord - self.get_domain().grid[dim][0])
                    / (self.get_domain().grid[dim][-1] - self.get_domain().grid[dim][0])
                )
            data = self.data.take(indices=N_c, axis=dim)
            other_dim = [i for i in self.all_dimensions() if i != dim]
            if rotate:
                other_dim.reverse()
                data = data.T

            # extent = (
            #     self.physical_domain.grid[other_dim[0]][0],
            #     self.physical_domain.grid[other_dim[0]][-1],
            #     self.physical_domain.grid[other_dim[1]][0],
            #     self.physical_domain.grid[other_dim[1]][-1],
            # )
            extent = (
                min(self.physical_domain.grid[other_dim[0]]),
                max(self.physical_domain.grid[other_dim[0]]),
                min(self.physical_domain.grid[other_dim[1]]),
                max(self.physical_domain.grid[other_dim[1]]),
            )
            x = self.physical_domain.grid[other_dim[0]]
            y = jnp.flip(self.physical_domain.grid[other_dim[1]])
            # y = self.physical_domain.grid[other_dim[1]]
            Nx = self.physical_domain.get_shape()[other_dim[0]]
            Ny = self.physical_domain.get_shape()[other_dim[1]]
            xi = np.linspace(x[0], x[-1], Nx)
            yi = np.linspace(y[0], y[-1], Ny)
            interp = RegularGridInterpolator((x, y), data, method="cubic")
            interp_data = np.array([[interp([[x_, y_]])[0] for x_ in xi] for y_ in yi])
            ims.append(
                ax.imshow(
                    interp_data,
                    interpolation=None,
                    extent=extent,
                )
            )
            ax.set_xlabel("$" + "xyz"[other_dim[0]] + "$")
            ax.set_ylabel("$" + "xyz"[other_dim[1]] + "$")
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
            vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in ims:
                im.set_norm(norm)
            name = params.get("name", self.name)
            no_cb = params.get("no_cb", False)
            if no_cb is False:
                name_color = params.get("name_color", "black")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(ims[0], cax=cax, label=name, orientation="vertical")
                cbar.ax.yaxis.label.set_color(name_color)
                ax.set_title("$" + "xyz"[dim] + " = " + "{:.2f}".format(coord) + "$")

            def save() -> None:
                if params.get("ax") is None:
                    assert fig is not None
                    fig.savefig(
                        (
                            self.plotting_dir
                            + "plot_3d_"
                            + "xyz"[dim]
                            + "_"
                            + ("no_cb_" if no_cb else "")
                            + self.name
                            + "_latest"
                            + self.plotting_format
                        ),
                        bbox_inches="tight",
                    )
                    fig.savefig(
                        (
                            self.plotting_dir
                            + "plot_3d_"
                            + "xyz"[dim]
                            + "_"
                            + ("no_cb_" if no_cb else "")
                            + self.name
                            + "_t_"
                            + "{:06}".format(self.time_step)
                            + self.plotting_format
                        ),
                        bbox_inches="tight",
                    )

            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()
            if params.get("ax") is None:
                del fig, ax

        except Exception as e:
            print("plot_3d_single failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_wavenumbers(self, normal_direction: int) -> None:
        try:
            assert (
                self.physical_domain.number_of_dimensions == 3
            ), "Only 3D supported for this plotting method."
            v_avg = cast(PhysicalField, self.definite_integral(normal_direction)).hat()
            name = self.name + "_avg"
            v_avg.set_time_step(self.time_step)
            fig = figure.Figure()
            ax_ = fig.subplots(1, 1)
            assert type(ax_) is Axes
            ims = []
            other_dim = [i for i in self.all_dimensions() if i != normal_direction]
            domain_hat = self.get_physical_domain().hat()
            ims.append(
                ax_.imshow(
                    np.fft.fftshift(
                        abs(v_avg.data.T), axes=(other_dim[0] if use_rfftn else None)
                    ),
                    interpolation=None,
                    extent=(
                        min(domain_hat.grid[other_dim[0]]),
                        max(domain_hat.grid[other_dim[0]]),
                        min(domain_hat.grid[other_dim[1]]),
                        max(domain_hat.grid[other_dim[1]]),
                    ),
                )
            )
            ax_.set_xlabel("$" + "xyz"[other_dim[0]] + "$")
            ax_.set_ylabel("$" + "xyz"[other_dim[1]] + "$")
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
            vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in ims:
                im.set_norm(norm)
            divider = make_axes_locatable(ax_)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(ims[0], cax=cax, label=name, orientation="vertical")

            def save() -> None:
                fig.savefig(
                    self.plotting_dir
                    + "plot_3d_"
                    + "y"
                    + "_"
                    + name
                    + "_latest"
                    + self.plotting_format,
                    bbox_inches="tight",
                )
                fig.savefig(
                    self.plotting_dir
                    + "plot_3d_"
                    + "y"
                    + "_"
                    + name
                    + "_t_"
                    + "{:06}".format(self.time_step)
                    + self.plotting_format,
                    bbox_inches="tight",
                )

            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()
            del fig, ax_
        except Exception as e:
            print("FourierField.plot_wavenumbers failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_isolines(
        self, normal_direction: int, isolines: Optional[List["jsd_float"]] = None
    ) -> None:
        try:
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
            cmap = colors.ListedColormap((("gray", 0.3), "white"))
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
            del fig, ax
        except Exception as e:
            print("plot_isolines failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_isosurfaces(
        self, iso_val: float = 0.6, plot_min_and_max: bool = True, **params: Any
    ) -> None:
        try:
            min_val = self.min()
            max_val = self.max()
            domain = self.get_physical_domain()
            name = params.get("name", self.name)
            grid = pv.RectilinearGrid(*domain.grid)
            wall_grid = pv.RectilinearGrid(*domain.grid)
            grid.point_data[name] = self.get_data().T.flatten()
            wall_grid.point_data[name] = domain.mgrid[1]
            wall_mesh = wall_grid.contour(1, -1)
            values = grid.point_data[name]
            other_values = params.get("other_values", [])
            if plot_min_and_max:
                # mesh = grid.contour([iso_val * min_val, iso_val * max_val], values)
                mesh = grid.contour(
                    [iso_val * max_val, iso_val * min_val]
                    + [val * max_val for val in other_values]
                    + [val * min_val for val in other_values],
                    values,
                )
            else:
                mesh = grid.contour(
                    [iso_val * max_val] + [val * max_val for val in other_values],
                    values,
                )

            interactive = params.get("interactive", False)
            try:
                font_size = int(matplotlib.rcParams["font.size"])
            except Exception:
                font_size = 18
            p = pv.Plotter(off_screen=(not interactive))
            # p.add_mesh(mesh.outline(), color="k")
            p.add_mesh(
                wall_mesh,
                opacity=params.get("opacity", 0.3),
            )
            p.add_mesh(
                mesh,
                opacity=params.get("opacity", 0.6),
                smooth_shading=True,
                cmap="viridis",
                scalar_bar_args={
                    "title_font_size": font_size,
                    "label_font_size": font_size,
                    "title": name,
                    "vertical": params.get("vertical_cbar", True),
                    "position_x": params.get("cbar_position_x", 0.85),
                    "position_y": params.get("cbar_position_y"),
                },
                # opacity=dist,
            )
            p.camera_position = "xy"
            p.camera.elevation = 20
            p.camera.roll = -0
            p.camera.azimuth = -45
            p.camera.zoom(0.9)
            p.show_axes()

            def save() -> None:
                out_name = (
                    self.plotting_dir
                    + "plot_isosurfaces_"
                    + self.name
                    + "_t_"
                    + "{:06}".format(self.time_step)
                    + self.plotting_format
                )
                if interactive:
                    p.show()
                else:
                    p.screenshot(out_name)
                    copyfile(
                        out_name,
                        self.plotting_dir
                        + "plot_isosurfaces_"
                        + self.name
                        + "_latest"
                        + self.plotting_format,
                    )

            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()
            p.close()
            p.deep_clean()
            del p
        except Exception as e:
            print("plot_isosurfaces failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def hat(self) -> FourierField:
        out = FourierField.FromField(self)
        out.time_step = self.time_step
        return out

    def diff(self, direction: int, order: int = 1) -> PhysicalField:
        name_suffix = "".join([["x", "y", "z"][direction] for _ in jnp.arange(order)])
        if self.physical_domain.is_periodic(direction):
            return self.hat().diff(direction, order).no_hat()
        else:
            return PhysicalField(
                self.physical_domain,
                self.physical_domain.diff(self.data, direction, order),
                self.name + "_" + name_suffix,
            )

    def integrate(
        self,
        direction: int,
        order: int = 1,
        bc_left: Optional[float] = None,
        bc_right: Optional[float] = None,
    ) -> PhysicalField:
        out_bc = self.physical_domain.integrate(
            self.data, direction, order, bc_left, bc_right
        )
        # assert type(out_bc) is 'jsd_array'
        return PhysicalField(self.physical_domain, out_bc, name=self.name + "_int")

    def definite_integral(
        self, direction: int
    ) -> Union[float, "jsd_array", PhysicalField]:
        def reduce_add_along_axis(arr: "jsd_array", axis: int) -> "jnp_array":
            # return np.add.reduce(arr, axis=axis)
            arr = jnp.moveaxis(arr, axis, 0)
            out_arr = jnp.array(functools.reduce(lambda a, b: a + b, arr))
            # assert type(out_arr) is 'jsd_array'
            return out_arr

        if not self.is_periodic(direction):
            int = self.integrate(direction, 1, bc_right=0.0)
            if self.number_of_dimensions() == 1:
                out: "jsd_float" = cast("jsd_float", int[0]) - cast(
                    "jsd_float", int[-1]
                )
                return cast(float, out)
            else:
                N = self.physical_domain.number_of_cells(direction)
                inds = [i for i in self.all_dimensions() if i != direction]
                physical_shape = tuple(
                    (
                        np.array(self.physical_domain.get_shape_aliasing())[
                            tuple(inds),
                        ]
                    ).tolist()
                )
                periodic_directions = tuple(
                    (
                        np.array(self.physical_domain.periodic_directions)[tuple(inds),]
                    ).tolist()
                )
                scale_factors = tuple(
                    (
                        np.array(self.physical_domain.scale_factors)[tuple(inds),]
                    ).tolist()
                )
                reduced_domain = PhysicalDomain.create(
                    physical_shape,
                    periodic_directions,
                    scale_factors=scale_factors,
                    aliasing=self.get_domain().aliasing,
                    dealias_nonperiodic=self.get_domain().dealias_nonperiodic,
                    physical_shape_passed=True,
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
                physical_shape = tuple(
                    (
                        np.array(self.physical_domain.get_shape_aliasing())[
                            tuple(inds),
                        ]
                    ).tolist()
                )
                periodic_directions = tuple(
                    (
                        np.array(self.physical_domain.periodic_directions)[tuple(inds),]
                    ).tolist()
                )
                scale_factors = tuple(
                    (
                        np.array(self.physical_domain.scale_factors)[tuple(inds),]
                    ).tolist()
                )
                reduced_domain = PhysicalDomain.create(
                    physical_shape,
                    periodic_directions,
                    scale_factors=scale_factors,
                    aliasing=self.get_domain().aliasing,
                    dealias_nonperiodic=self.get_domain().dealias_nonperiodic,
                    physical_shape_passed=True,
                )
                data = (
                    self.physical_domain.scale_factors[direction]
                    / N
                    * reduce_add_along_axis(self.data, direction)
                )
                return PhysicalField(reduced_domain, data)


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
    def FromRandom(
        cls,
        domain: PhysicalDomain,
        seed: "jsd_float" = 0.0,
        energy_norm: float = 1.0,
        name: str = "field",
    ) -> FourierField:
        """Construct a random field depending on the independent variables described by domain."""
        return PhysicalField.FromRandom(domain, seed, energy_norm, name).hat()

    @classmethod
    def FromWhiteNoise(
        cls,
        domain: PhysicalDomain,
        energy_norm: float = 1.0,
        name: str = "field",
        seed: float = 37,
    ) -> FourierField:
        return cls.FromRandom(domain, seed, energy_norm, name).filter()

    @classmethod
    def Zeros(cls, domain: PhysicalDomain, name: str = "field") -> Self:
        data: "jnp_array" = jnp.zeros(domain.get_shape())
        return cls(domain, data, name)

    def get_domain(self) -> FourierDomain:
        return self.fourier_domain

    def get_physical_domain(self) -> PhysicalDomain:
        return self.physical_domain

    def __add__(self, other: Union[Self, jnp.ndarray]) -> FourierField:
        assert not isinstance(
            other, PhysicalField
        ), "Attempted to add a Fourier Field and a PhysicalField."
        if isinstance(other, FourierField):
            if self.activate_jit_:
                new_name = ""
            else:
                if other.name[0] == "-":
                    new_name = self.name + " - " + other.name[1:]
                else:
                    new_name = self.name + " + " + other.name
            ret = FourierField(
                self.physical_domain, self.data + other.data, name=new_name
            )
            ret.time_step = self.time_step
        else:
            new_name = self.name
            ret = FourierField(self.physical_domain, self.data + other, name=new_name)
            ret.time_step = self.time_step
        return ret

    def __sub__(self, other: Union[Self, jnp.ndarray]) -> FourierField:
        return self + other * (-1.0)  # type: ignore

    def __mul__(self, other: Union[Self, jnp.ndarray, "jsd_float"]) -> FourierField:
        if isinstance(other, Field):
            assert isinstance(
                other, FourierField
            ), "Attempted to multiply a FourierField and a PhysicalField."

        if isinstance(other, FourierField):
            raise Exception(
                "multiplication of FourierField and FourierField detected - this should probably be done in physical space."
            )
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

    def __truediv__(self, other: "jsd_float") -> FourierField:
        out = self.data / other
        return FourierField(self.physical_domain, out, name=self.name)

    def shift(self, value: "jsd_float") -> FourierField:
        out_field = self.data + value
        return FourierField(self.get_physical_domain(), out_field, name=self.name)

    @classmethod
    def FromField(cls, field: PhysicalField) -> FourierField:
        out = cls(field.physical_domain, field.data, field.name + "_hat")
        out.physical_domain = field.physical_domain
        out.fourier_domain = field.physical_domain.hat()
        out.data = out.physical_domain.field_hat(field.data)
        return out

    def field_2d(self, direction: int, wavenumber: int = 0) -> FourierField:
        N = self.data.shape[direction]

        def get_data(wn: int) -> "jnp_array":
            u_hat_const_data_0 = jnp.take(
                self.data, indices=jnp.arange(wn, wn + 1), axis=direction
            )
            u_hat_const_data_pre = jnp.zeros_like(
                jnp.take(self.data, indices=jnp.arange(0, wn), axis=direction)
            )
            u_hat_const_data_post = jnp.zeros_like(
                jnp.take(self.data, indices=jnp.arange(wn + 1, N), axis=direction)
            )
            u_hat_const_data = jnp.concatenate(
                [u_hat_const_data_pre, u_hat_const_data_0, u_hat_const_data_post],
                axis=direction,
            )
            return u_hat_const_data

        if wavenumber == 0:
            u_hat_const_data = get_data(wavenumber)
        else:
            u_hat_const_data = get_data(wavenumber) + get_data(N - wavenumber)
        return FourierField(self.get_physical_domain(), u_hat_const_data)

    def energy_2d(self, direction: int) -> float:
        return self.field_2d(direction).no_hat().energy()

    def project_onto_domain(self, domain: PhysicalDomain) -> FourierField:
        out_data = self.get_domain().project_onto_domain(domain, self.data)
        return FourierField(domain, out_data, name=self.name)

    def filter(self) -> FourierField:
        out_data = self.get_domain().filter_field(self.data)
        return FourierField(self.get_physical_domain(), out_data, name=self.name)

    def filter_fourier(self) -> FourierField:
        out_data = self.get_domain().filter_field_fourier_only(self.data)
        return FourierField(self.get_physical_domain(), out_data, name=self.name)

    def filter_nonfourier(self) -> FourierField:
        out_data = self.get_domain().filter_field_nonfourier_only(self.data)
        return FourierField(self.get_physical_domain(), out_data, name=self.name)

    def number_of_dofs_aliasing(self) -> int:
        return int(math.prod(self.get_physical_domain().shape))

    def normalize_by_max_value(self) -> Self:
        raise Exception(
            "This is not supported for Fourier Fields. Transform to PhysicalField, normalize, and transform back to FourierField instead."
        )

    def diff(self, direction: int, order: int = 1) -> FourierField:
        domain: FourierDomain = self.get_domain()
        out_field: "jnp_array" = domain.diff(self.data, direction, order)
        return FourierField(
            self.physical_domain,
            out_field,
            name=self.name + "_" + "xyz"[direction],
        )

    def integrate(
        self,
        direction: int,
        order: int = 1,
        bc_right: Optional[float] = None,
        bc_left: Optional[float] = None,
    ) -> FourierField:
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
            out_field_ = self.physical_domain.integrate(
                self.data, direction, order, bc_right=bc_right, bc_left=bc_left
            )
            # assert type(out_field_) is 'jsd_array'
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

    def definite_integral(
        self, direction: int
    ) -> Union["jsd_float", "jsd_array", PhysicalField]:
        raise NotImplementedError()

    def update_boundary_conditions(self) -> None:
        """Divide field by the absolute value of its maximum, unless it is
        very small (this prevents divide-by-zero issues)."""
        raise NotImplementedError()

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

    def solve_poisson(self, mat: Optional["np_complex_array"] = None) -> FourierField:
        assert len(self.all_dimensions()) == 3, "Only 3d implemented currently."
        assert (
            len(self.all_nonperiodic_dimensions()) <= 1
        ), "Poisson solution not implemented for the general case."
        rhs_hat = self.data
        if type(mat) == NoneType:
            mat_ = self.assemble_poisson_matrix()
        else:
            assert mat is not None
            mat_ = mat
        field = rhs_hat
        out_field = jnp.pad(
            jnp.einsum("ijkl,ilj->ikj", mat_, field),
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

    def no_hat(self) -> PhysicalField:
        out = self.fourier_domain.field_no_hat(self.data)
        out_field = PhysicalField(
            self.physical_domain, out, name=(self.name).replace("_hat", "")
        )
        out_field.time_step = self.time_step
        return out_field

    def reconstruct_from_wavenumbers(
        self, fn: Callable[[int, int], "jnp_array"], vectorize: bool = False
    ) -> FourierField:
        if vectorize:
            print("vectorisation not implemented yet, using unvectorized version")
        assert self.number_of_dimensions() == 3, "Only 3D implemented."
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

    def get_streak_scales(self) -> "Tuple[float, float]":
        vel_x_0_hat = self.field_2d(0)
        vel_x_0 = vel_x_0_hat.no_hat()
        vel_x_0.set_time_step(self.get_time_step())
        vel_x_0.set_name("vel_x_streaks")

        # Nx, Ny, Nz = vel_x_0.get_data().shape
        # domain = vel_x_0.physical_domain
        # coarse_domain = PhysicalDomain.create(
        #     (Nx, Ny, 20), domain.periodic_directions, domain.scale_factors
        # )

        # max_inds = np.unravel_index(
        #     vel_x_0.get_data().argmax(axis=None), vel_x_0.get_data().shape
        # )
        max_inds = np.unravel_index(
            (vel_x_0.get_data() ** 2).argmax(axis=None), vel_x_0.get_data().shape
        )
        # x_max = max_inds[0] / Nx * vel_x_0.physical_domain.grid[0][-1]
        # z_max = max_inds[2] / Nz * vel_x_0.physical_domain.grid[2][-1]
        lambda_y = 1 - abs(vel_x_0.physical_domain.grid[1][max_inds[1]])
        # print("max_inds", max_inds)
        # print("lambda_y:", lambda_y)
        # max_inds_hat = np.unravel_index(
        #     abs(vel_x_0_hat.get_data()[:, :, 1:]).argmax(axis=None),
        #     vel_x_0_hat.get_data()[:, :, 1:].shape,
        # )
        v = vel_x_0_hat
        max_inds_hat = np.unravel_index(
            abs(v.get_data()[:, :, 1:]).argmax(axis=None),
            vel_x_0_hat.get_data()[:, :, 1:].shape,
        )

        lambda_z = abs(
            2 * np.pi / vel_x_0_hat.fourier_domain.grid[2][max_inds_hat[2] + 1]
        )
        # slice_field_data = vel_x_0[0, max_inds[1], :]
        # slice_field_data_sq = (vel_x_0**2)[0, max_inds[1], :]
        # slice_grid_data = self.get_physical_domain().grid[2]
        # fig = figure.Figure()
        # ax = fig.subplots(1, 1)
        # ax.plot(slice_grid_data, slice_field_data)
        # ax.plot(slice_grid_data, slice_field_data_sq)
        # fig.savefig(self.plotting_dir + "/plot_" + str(self.get_time_step()) + ".png")
        # self.plot_3d(0, coord=x_max)

        return (cast("float", lambda_y), cast("float", lambda_z))

    def plot_3d(
        self,
        direction: Optional[int] = None,
        coord: Optional[float] = None,
        rotate: bool = False,
        **params: Any,
    ) -> None:
        try:
            if direction is not None:
                self.plot_3d_single(direction, coord, rotate, **params)
            else:
                assert (
                    self.physical_domain.number_of_dimensions == 3
                ), "Only 3D supported for this plotting method."
                try:
                    fig = figure.Figure(layout="constrained")
                except Exception:
                    fig = figure.Figure()
                base_len = 100
                grd = (base_len, base_len)
                lx = self.get_domain().get_shape()[0]
                ly = self.get_domain().get_shape()[1]
                lz = self.get_domain().get_shape()[2]
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
                    data_shape = self.data.shape
                    if coord is None:
                        N_c = (data_shape[dim] - 1) // 2
                        coord = (
                            N_c
                            / data_shape[dim]
                            * (
                                self.get_domain().grid[dim][-1]
                                - self.get_domain().grid[dim][0]
                            )
                        )
                    else:
                        N_c = int(
                            (data_shape[dim] - 1)
                            * (coord - self.get_domain().grid[dim][0])
                            / (
                                self.get_domain().grid[dim][-1]
                                - self.get_domain().grid[dim][0]
                            )
                        )
                    other_dim = [i for i in self.all_dimensions() if i != dim]
                    ims.append(
                        ax[dim].imshow(
                            np.fft.fftshift(abs(self.data.take(indices=N_c, axis=dim))),
                            interpolation=None,
                            extent=(
                                min(self.get_domain().grid[other_dim[0]]),
                                max(self.get_domain().grid[other_dim[0]]),
                                min(self.get_domain().grid[other_dim[1]]),
                                max(self.get_domain().grid[other_dim[1]]),
                            ),
                        )
                    )
                    ax[dim].set_xlabel("$" + "xyz"[other_dim[1]] + "$")
                    ax[dim].set_ylabel("$" + "xyz"[other_dim[0]] + "$")
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
                vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in ims:
                    im.set_norm(norm)
                name = params.get("name", self.name)
                name_color = params.get("name_color", "black")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(ims[0], cax=cax, label=name, orientation="vertical")
                cbar.ax.yaxis.label.set_color(name_color)
                assert coord is not None
                ax[dim].set_title(
                    "$" + "xyz"[dim] + " = " + "{:.2f}".format(coord) + "$"
                )

                def save() -> None:
                    fig.savefig(
                        self.plotting_dir
                        + "plot_3d_"
                        + self.name
                        + "_latest"
                        + self.plotting_format,
                        bbox_inches="tight",
                    )
                    fig.savefig(
                        self.plotting_dir
                        + "plot_3d_"
                        + self.name
                        + "_t_"
                        + "{:06}".format(self.time_step)
                        + self.plotting_format,
                        bbox_inches="tight",
                    )

                try:
                    save()
                except FileNotFoundError:
                    Field.initialize(False)
                    save()
                del fig, ax
        except Exception as e:
            print("FourierField.plot_3d failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")

    def plot_3d_single(
        self,
        dim: int,
        coord: Optional[float],
        rotate: bool = False,
        **params: Any,
    ) -> None:
        try:
            assert (
                self.physical_domain.number_of_dimensions == 3
            ), "Only 3D supported for this plotting method."
            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ims = []
            data_shape = self.data.shape
            if coord is None:
                N_c = (data_shape[dim] - 1) // 2
                coord = (
                    N_c
                    / data_shape[dim]
                    * (self.get_domain().grid[dim][-1] - self.get_domain().grid[dim][0])
                )
            else:
                N_c = int(
                    (data_shape[dim] - 1)
                    * (coord - self.get_domain().grid[dim][0])
                    / (self.get_domain().grid[dim][-1] - self.get_domain().grid[dim][0])
                )
            data = self.data.take(indices=N_c, axis=dim).T
            other_dim = [i for i in self.all_dimensions() if i != dim]
            if rotate:
                other_dim.reverse()
                data = data.T
            ims.append(
                ax.imshow(
                    np.fft.fftshift(abs(data)),
                    interpolation=None,
                    extent=(
                        min(self.get_domain().grid[other_dim[0]]),
                        max(self.get_domain().grid[other_dim[0]]),
                        min(self.get_domain().grid[other_dim[1]]),
                        max(self.get_domain().grid[other_dim[1]]),
                    ),
                )
            )
            ax.set_xlabel("$" + "xyz"[other_dim[1]] + "$")
            ax.set_ylabel("$" + "xyz"[other_dim[0]] + "$")
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in ims)  # type: ignore[union-attr]
            vmax = max(image.get_array().max() for image in ims)  # type: ignore[union-attr]
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in ims:
                im.set_norm(norm)
            name = params.get("name", self.name)
            name_color = params.get("name_color", "black")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(ims[0], cax=cax, label=name, orientation="vertical")
            cbar.ax.yaxis.label.set_color(name_color)
            ax.set_title("$" + "xyz"[dim] + " = " + "{:.2f}".format(coord) + "$")

            def save() -> None:
                fig.savefig(
                    self.plotting_dir
                    + "plot_3d_"
                    + "xyz"[dim]
                    + "_"
                    + self.name
                    + "_latest"
                    + self.plotting_format,
                    bbox_inches="tight",
                )
                fig.savefig(
                    self.plotting_dir
                    + "plot_3d_"
                    + "xyz"[dim]
                    + "_"
                    + self.name
                    + "_t_"
                    + "{:06}".format(self.time_step)
                    + self.plotting_format,
                    bbox_inches="tight",
                )

            try:
                save()
            except FileNotFoundError:
                Field.initialize(False)
                save()
            del fig, ax
        except Exception as e:
            print("FourierField.plot_3d_single failed with the following exception:")
            print(e)
            print("ignoring this and carrying on.")


class FourierFieldSlice(FourierField):
    def __init__(
        self,
        domain: FourierDomain,
        non_periodic_direction: int,
        data: "jnp_array",
        name: str = "field_hat_slice",
        *ks: int,
        **params: Any,
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

    def all_periodic_dimensions(self) -> List[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if d not in self.all_nonperiodic_dimensions()
        ]

    def all_nonperiodic_dimensions(self) -> List[int]:
        return [self.non_periodic_direction]

    def diff(self, direction: int, order: int = 1) -> "FourierFieldSlice":
        if direction in self.all_periodic_dimensions():
            diff_array = (1j * self.ks[direction]) ** order
            out_field = diff_array * self.data
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

    def integrate(
        self,
        direction: int,
        order: int = 1,
        _: Optional["jsd_array"] = None,
        __: Optional["jsd_array"] = None,
    ) -> "FourierFieldSlice":
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

    def assemble_poisson_matrix(self) -> "np_complex_array":
        y_mat = self.get_cheb_mat_2_homogeneous_dirichlet(0)
        n = y_mat.shape[0]
        factor = np.zeros_like(self.ks[0])
        for direction in self.all_periodic_dimensions():
            factor += (1j * self.ks[direction]) ** 2

        I = np.eye(n)
        mat = factor * I + y_mat
        mat_inv = np.linalg.inv(mat)
        return mat_inv

    def solve_poisson(
        self, mat: Optional["np_complex_array"] = None
    ) -> FourierFieldSlice:
        if type(mat) == NoneType:
            mat_inv = self.assemble_poisson_matrix()
        else:
            assert mat is not None
            k1 = self.ks_int[self.all_periodic_dimensions()[0]]
            k2 = self.ks_int[self.all_periodic_dimensions()[1]]
            mat_inv = mat[k1, k2, :, :]
        rhs_hat = self.data
        out_field = mat_inv @ rhs_hat
        out_fourier = FourierFieldSlice(
            self.fourier_domain,
            self.non_periodic_direction,
            jnp.array(out_field),
            self.name + "_poisson",
            *self.ks_raw,
            ks_int=self.ks_int[jnp.array(self.all_periodic_dimensions())],
        )
        return out_fourier

    def update_boundary_conditions(self) -> None:
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
        return self + other * (-1.0)  # type: ignore

    def __mul__(
        self, other: Union[Self, jnp.ndarray, "jsd_float"]
    ) -> FourierFieldSlice:
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

    def __truediv__(self, other: "jsd_float") -> FourierFieldSlice:
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

    def shift(self, value: "jsd_float") -> FourierFieldSlice:
        out_field = self.data + value
        return FourierFieldSlice(
            self.fourier_domain, self.non_periodic_direction, out_field, name=self.name
        )

#!/usr/bin/env python3

from __future__ import annotations
import jax._src.typing as jt
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Iterable, Optional, Sequence, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from jax_spectral_dns.field import PhysicalField, FourierField, VectorField


np_float_array = npt.NDArray[np.float64]
np_complex_array = npt.NDArray[np.complex64]
jsd_float = Union[float, np.float64, jnp.float64]
jsd_complex = Union[jsd_float, complex, np.complex64, jnp.complex64]
jnp_array = jt.Array
jsd_array = jt.ArrayLike
np_jnp_array = Union[np_float_array, np_complex_array, jnp_array]
Vel_fn_type = Callable[[Union[list[jsd_float], tuple[jsd_float,...], np_jnp_array]], jsd_float]
parameter_type = tuple[jnp_array, ...]
input_type = Any

if TYPE_CHECKING:
    AnyScalarField = Union[PhysicalField, FourierField]
    AnyVectorField = Union[VectorField[PhysicalField], VectorField[FourierField]]
    AnyField = Union[AnyVectorField, AnyScalarField]
    AnyFieldList = Union[List[PhysicalField], List[FourierField], List[VectorField[PhysicalField]], List[VectorField[FourierField]]]

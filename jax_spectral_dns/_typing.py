#!/usr/bin/env python3

from __future__ import annotations
import jax._src.typing as jt
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    List,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from jax_spectral_dns.field import PhysicalField, FourierField, VectorField


np_float_array = npt.NDArray[np.float64]
np_complex_array = npt.NDArray[np.complex128]
jsd_float = Union[float, np.float64, jnp.float64]
jsd_complex = Union[jsd_float, complex, np.complex128, jnp.complex128]
jnp_array = jt.Array
jsd_array = jt.ArrayLike
np_jnp_array = Union[np_float_array, np_complex_array, jnp_array]
Vel_fn_type = Callable[
    [Union[List[jsd_float], Tuple[jsd_float, ...], np_jnp_array]], jsd_float
]
parameter_type = Tuple[jnp_array, ...]

if TYPE_CHECKING:
    AnyScalarField = Union[PhysicalField, FourierField]
    AnyVectorField = Union[VectorField[PhysicalField], VectorField[FourierField]]
    AnyField = Union[AnyVectorField, AnyScalarField]
    AnyFieldList = Union[
        List[PhysicalField],
        List[FourierField],
        List[VectorField[PhysicalField]],
        List[VectorField[FourierField]],
    ]
    AnyFieldSequence = Union[
        Sequence[PhysicalField],
        Sequence[FourierField],
        Sequence[VectorField[PhysicalField]],
        Sequence[VectorField[FourierField]],
    ]
    pseudo_2d_perturbation_return_type = Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        VectorField[PhysicalField],
    ]

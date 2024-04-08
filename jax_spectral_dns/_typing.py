#!/usr/bin/env python3

import jax._src.typing as jt
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Callable, Iterable, Optional, Sequence, Union

from jax_spectral_dns.field import Field, FourierField, PhysicalField, VectorField
# from typing_extensions import Self


np_float_array = npt.NDArray[np.float64]
np_complex_array = npt.NDArray[np.complex64]
jsd_float = Union[float, np.float64, jnp.float64]
jsd_complex = Union[jsd_float, complex, np.complex64, jnp.complex64]
jnp_array = jt.Array
jsd_array = jt.ArrayLike
np_jnp_array = Union[np_float_array, np_complex_array, jnp_array]
AnyField = Union[Field, VectorField, PhysicalField, FourierField]

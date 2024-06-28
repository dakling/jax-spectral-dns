#!/usr/bin/env python3

from __future__ import annotations

import jax
import numpy as np
import sys

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import PhysicalField, FourierField, VectorField
from jax_spectral_dns.main import get_args_from_yaml_string

args = dict()

if len(sys.argv) > 1:
    args = get_args_from_yaml_string(" ".join(sys.argv[1:]))

# adapt this to your case
domain = PhysicalDomain.create(
    (64, 129, 32), (True, False, True), (2.0 * np.pi, 1.0, 1.0 * np.pi)
)

iteration = args.get("iteration")
if iteration is not None:
    u = VectorField.FromFile(domain, "velocity_latest_" + str(iteration))
else:
    u = VectorField.FromFile(domain, "velocity_latest")

component = args.get("component", 0)
u[component].plot_isosurfaces(args.get("iso_val", 0.6), interactive=True)

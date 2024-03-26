#!/usr/bin/env python3

import jax
import warnings
# warnings.filterwarnings('error')
# import jax.profiler

jax.config.update("jax_enable_x64", True)
import os
import multiprocessing

max_devices = 1
# max_devices = 6
# max_devices = 1e10
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    min(multiprocessing.cpu_count(), max_devices)
)

import sys
from jax_spectral_dns.examples import *


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise Exception("Please indicate a function from examples.py that should be run.")
    else:
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)

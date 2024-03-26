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
from examples import *


def main():
    pass
    # solve_heat_eq_1D()
    # optimize_heat_eq_1D()
    # run_jimenez_1990()
    # run_transient_growth(600, 2.0)
    # run_pseudo_2d_perturbation()
    # run_optimization_pseudo_2d_perturbation()
    # run_optimization_transient_growth()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        globals()[sys.argv[1]]()
    else:
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)

# jax.profiler.save_device_memory_profile("memory.prof")

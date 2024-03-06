#!/usr/bin/env python3

import jax
import warnings
warnings.filterwarnings('ignore')
import jax.profiler

jax.config.update("jax_enable_x64", True)
import os
import multiprocessing

max_devices = 1
# max_devices = 6
# max_devices = 1e10
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    min(multiprocessing.cpu_count(), max_devices)
)

# from importlib import reload
import sys

# try:
#     reload(sys.modules["heat_eq"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load heat_eq")
from heat_eq import optimize_heat_eq_1D, solve_heat_eq_1D

# try:
#     reload(sys.modules["test_project"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load test_project")
from test_project import TestProject

# try:
#     reload(sys.modules["examples"])
# except:
#     if hasattr(sys, 'ps1'):
#         print("Unable to load examples")
from examples import *


def main():
    # pass
    # solve_heat_eq_1D()
    # optimize_heat_eq_1D()
    # run_jimenez_1990()
    # run_transient_growth(600, 2.0)
    # run_pseudo_2d_perturbation()
    # run_optimization_pseudo_2d_perturbation()
    # run_optimization_transient_growth()

    # run tests
    tp = TestProject()
    tp.test_navier_stokes_laminar()
    # tp.test_perturbation_laminar()
    # tp.test_2d_growth()
    # tp.test_linear_stability()
    raise Exception("break")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        globals()[sys.argv[1]]()
    else:
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)

jax.profiler.save_device_memory_profile("memory.prof")

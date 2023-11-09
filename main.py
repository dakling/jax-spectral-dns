#!/usr/bin/env python3

import jax

jax.config.update("jax_enable_x64", True)

from importlib import reload
import sys

try:
    reload(sys.modules["test_project"])
except:
    if hasattr(sys, 'ps1'):
        print("Unable to load test_project")
from test_project import TestProject

try:
    reload(sys.modules["examples"])
except:
    if hasattr(sys, 'ps1'):
        print("Unable to load examples")
from examples import *


def main():
    pass
    # run_jimenez_1990()
    # run_transient_growth()
    # run_pseudo_2d_pertubation(Re=5.5e3, Ny=240)

    # run tests
    # tp = TestProject()
    # tp.test_navier_stokes_laminar()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        globals()[sys.argv[1]]()
    else:
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)

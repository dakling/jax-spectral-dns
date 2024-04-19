#!/usr/bin/env python3

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
# import warnings
# warnings.filterwarnings('error')
# import jax.profiler
import logging

import os
import multiprocessing
from jax_spectral_dns.equation import print_verb

logging.getLogger("jax").setLevel(logging.WARNING)

# max_devices = 1
max_devices = 6
# max_devices = 1e10
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    min(multiprocessing.cpu_count(), max_devices)
)

import sys
from jax_spectral_dns.examples import *


def print_welcome() -> None:
    vl = 0
    print("")
    print("")
    print_verb("#############################", verbosity_level=vl)
    print_verb("# Starting jax-spectral-dns #", verbosity_level=vl, notify=True)
    print_verb("#############################", verbosity_level=vl)
    print("")
    print("")


def print_goodbye() -> None:
    vl = 0
    print("")
    print("")
    print_verb("######################", verbosity_level=vl)
    print_verb("# End of run reached #", verbosity_level=vl, notify=True)
    print_verb("######################", verbosity_level=vl)
    print("")
    print("")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception(
            "Please indicate a function from examples.py that should be run."
        )
    else:
        print_welcome()
        args = sys.argv[2:]
        globals()[sys.argv[1]](*args)
        print_goodbye()

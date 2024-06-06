#!/usr/bin/env python3
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
# import warnings
# warnings.filterwarnings('error')
import logging
import argparse

import os
import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)
os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)

logging.getLogger("jax").setLevel(logging.WARNING)

# max_devices = 1
# max_devices = 3
# max_devices = 1e10
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     min(multiprocessing.cpu_count(), max_devices)
# )

verbose = int(os.environ.get("JAX_SPECTRAL_DNS_VERBOSITY_LEVEL", 1))

import sys
from jax_spectral_dns.examples import *
from jax_spectral_dns.equation import print_verb, Equation

Equation.verbosity_level = verbose


def print_welcome() -> None:
    vl = 0
    print("")
    print("")
    print_verb("#############################", verbosity_level=vl)
    print_verb("# Starting jax-spectral-dns #", verbosity_level=vl, notify=True)
    print_verb("#############################", verbosity_level=vl)
    print("")
    print_verb("jax.local_devices(): ", jax.local_devices(), verbosity_level=2)
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


def print_failure() -> None:
    vl = 0
    print("")
    print("")
    print_verb("##############", verbosity_level=vl)
    print_verb("# Run failed #", verbosity_level=vl, notify=True)
    print_verb("##############", verbosity_level=vl)
    print("")
    print("")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception(
            "Please indicate a function from examples.py that should be run."
        )
    else:
        print_welcome()
        Equation.verbosity_level = verbose
        try:
            args = sys.argv[2:]
            globals()[sys.argv[1]](*args)
        except Exception as e:
            print(e)
            print_failure()
            raise e
        print_goodbye()

#!/usr/bin/env python3
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
# import warnings
# warnings.filterwarnings('error')
import logging

import os
import yaml
from typing import Any

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


def get_args_from_yaml_file() -> Any:
    with open("simulation_settings.yml", "r") as file:
        args = yaml.safe_load(file)
    return args


def get_args_from_yaml_string(string: str) -> Any:
    string_ = string.replace(" ", "\n").replace("=", ": ")
    args = yaml.safe_load(string_)
    return args


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception(
            "Please indicate a function from examples.py that should be run."
        )
    else:
        print_welcome()
        Equation.verbosity_level = verbose
        try:
            func_name = sys.argv[1]
            try:
                args = get_args_from_yaml_file()
                print_verb("reading simulation_settings.yml")
                if len(sys.argv) > 2:
                    print_verb(
                        "WARNING: command line arguments are ignored in favor of simulation_settings.yml"
                    )
            except FileNotFoundError:
                print_verb(
                    "WARNING: file simulation_settings.yml not found. Reading arguments from command line, which is discouraged."
                )
                try:
                    args = get_args_from_yaml_string(" ".join(sys.argv[2:]))
                    with open("simulation_settings_.yml", "w") as file:
                        yaml.dump(args, file)
                        print_verb(
                            "writing out arguments to file simulation_settings_.yml. Rename this file to simulation_settings.yml to make sure it is read during the next run of jax-spectral-dns."
                        )
                except yaml.YAMLError as e:
                    print_verb("could not parse command line arguments.")
                    print_verb(
                        "usage:   python main.py <function name> option1=value1 option2=value2 ..."
                    )
                    print_verb(
                        "example: python main.py run_ld_2021_dual end_time=1.0 max_cfl=0.7 ..."
                    )
                    raise e
            if type(args) is not NoneType:
                globals()[func_name](**args)
            else:
                globals()[func_name]()
        except Exception as e:
            print(e)
            print_failure()
            raise e
        print_goodbye()

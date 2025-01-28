#!/usr/bin/env python3


import os
import sys

from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import glob

from typing import Iterable, List, TypeVar, Tuple
import pyvista as pv
import numpy as np
import h5py
import matplotlib
from matplotlib import figure
from matplotlib.axes import Axes
import jax.numpy as jnp
from jax_spectral_dns.cheb import cheby
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, VectorField, PhysicalField, FourierField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

matplotlib.set_loglevel("error")

from PIL import Image

# matplotlib.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']
font_size = 18
matplotlib.rcParams.update(
    {
        "font.size": font_size,
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)
pv.global_theme.font.size = font_size
pv.global_theme.font.title_size = font_size
pv.global_theme.font.label_size = font_size


# STORE_PREFIX = "/store/DAMTP/dsk34"
STORE_PREFIX = "/data/septal/dsk34"
HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)

args = get_args_from_yaml_file(HOME_DIR_BASE + "/simulation_settings.yml")
print(args)


def get_domain():
    Lx_over_pi = args.get("Lx_over_pi", 2.0)
    Lz_over_pi = args.get("Lz_over_pi", 1.0)
    Nx = args.get("Nx", 64)
    Ny = args.get("Ny", 129)
    Nz = args.get("Nz", 80)
    return PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
        aliasing=3 / 2,
        dealias_nonperiodic=False,
    )


def post_process():
    domain = get_domain()

    slice_domain = PhysicalDomain.create(
        (domain.get_shape_aliasing()[1],),
        (False,),
        scale_factors=(1.0,),
        aliasing=1,
    )
    avg_slice = PhysicalField.FromFile(
        slice_domain, "average_velocity", name="average_velocity_x"
    )
    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data = np.moveaxis(
        np.tile(np.tile(avg_slice.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)),
        1,
        2,
    )
    avg_ = PhysicalField(domain, jnp.asarray(u_data))
    t = 0
    try:
        try:
            with open(Field.field_dir + "/streak_data.txt", "r") as streak_file:
                last_line = streak_file.readlines()[-1]
                t_0 = int(last_line.split(",")[0])
        except Exception as e:
            print(e)
            t_0 = 0
            print(t_0)
        for fl in sorted(
            glob.glob("trajectory_yz_20*", root_dir="./fields"),
            key=lambda f: os.path.getmtime("fields/" + f),
        ):
            print("doing file " + fl)
            with h5py.File("fields/" + fl, "r") as f:
                velocity_yz_trajectory = f["trajectory_yz"]
                n_steps = velocity_yz_trajectory.shape[0]
                for i in range(n_steps):
                    if t >= t_0:
                        vel_yz = velocity_yz_trajectory[i]

                        u_data = np.tile(vel_yz, reps=(nx, 1, 1))
                        vel_yz_3d = (
                            PhysicalField(
                                domain,
                                np.tile(vel_yz, reps=(nx, 1, 1)),
                                name="vel_pert",
                            )
                            - avg_
                        )
                        vel_yz_3d.set_time_step(t)
                        vel_yz_3d.set_name("velocity_yz")
                        vel_yz_3d.plot_3d(0)
                        vel_yz_3d.plot_3d(2)

                        # Re_tau = args.get("Re_tau", 180)
                        lambda_y, lambda_z = vel_yz_3d.hat().get_streak_scales()
                        # print("lambda_y:", lambda_y)
                        # print("lambda_y+:", lambda_y * Re_tau)
                        # print("lambda_z:", lambda_z)
                        # print("lambda_z+:", lambda_z * Re_tau)
                        streak_amplitude = max(abs(vel_yz_3d.get_data().flatten()))
                        # print("streak inf norm", streak_amplitude)
                        # ts.append(time_step)
                        # lambda_y_s.append(lambda_y)
                        # lambda_z_s.append(lambda_z)
                        # streak_inf_norm_s.append(streak_amplitude)

                        with open(
                            Field.field_dir + "/streak_data.txt", "a"
                        ) as streak_file:
                            streak_file.write(
                                str(t)
                                + ", "
                                + str(lambda_y)
                                + ", "
                                + str(lambda_z)
                                + ", "
                                + str(streak_amplitude)
                                + "\n"
                            )
                    t += 1

        # for vel_yz in vel_yz_s:
        #     u_data = np.tile(vel_yz, reps=(nx, 1, 1))
        #     vel_yz_3d = (
        #         PhysicalField(domain, np.tile(vel_yz, reps=(nx, 1, 1)), name="vel_pert")
        #         - avg_[0]
        #     )
        #     vel_yz_3d.set_time_step(time_step)
        #     vel_yz_3d.set_name("velocity_yz")
        #     vel_yz_3d.plot_3d(0)
        #     vel_yz_3d.plot_3d(2)

        #     Re_tau = args.get("Re_tau", 180)
        #     lambda_y, lambda_z = vel_yz_3d.hat().get_streak_scales()
        #     # print("lambda_y:", lambda_y)
        #     # print("lambda_y+:", lambda_y * Re_tau)
        #     # print("lambda_z:", lambda_z)
        #     # print("lambda_z+:", lambda_z * Re_tau)
        #     streak_amplitude = max(abs(vel_yz_3d.get_data().flatten()))
        #     # print("streak inf norm", streak_amplitude)
        #     # ts.append(time_step)
        #     # lambda_y_s.append(lambda_y)
        #     # lambda_z_s.append(lambda_z)
        #     # streak_inf_norm_s.append(streak_amplitude)
        #     out.append([time_step, lambda_y, lambda_z, streak_amplitude])
        #     time_step += 1
        # # out_arr = np.array([[ts[k], lambda_y_s[k], lambda_z_s[k], streak_inf_norm_s[k]] for k in range(len(ts))])
        # out_arr = np.array(out)
        # print(out_arr.shape)
        # with open(Field.field_dir + "/streak_data.txt", "w") as f:
        #     np.savetxt(f, out_arr, delimiter=",")

    except Exception as e:
        # print(e)
        raise e


post_process()

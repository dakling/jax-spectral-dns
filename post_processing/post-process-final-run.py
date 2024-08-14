#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np
import h5py
from matplotlib import figure
from matplotlib.axes import Axes
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, VectorField, PhysicalField, FourierField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation


def get_domain(shape, Lx_over_pi: float, Lz_over_pi: float):
    return PhysicalDomain.create(
        shape,
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
    )


def post_process(
    file: str,
    end_time: float,
    Lx_over_pi: float,
    Lz_over_pi: float,
    time_step_0: int = 0,
) -> None:
    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        n_steps = velocity_trajectory.shape[0]
        domain = get_domain(velocity_trajectory.shape[2:], Lx_over_pi, Lz_over_pi)

        ts = []
        energy_t = []
        energy_x_2d = []
        energy_x_3d = []
        amplitude_t = []
        amplitude_x_2d_t = []
        amplitude_3d_t = []
        # prod = []
        # diss = []
        print("preparing")
        for j in range(n_steps):
            print("preparing, step", j + 1, "of", n_steps)
            vel_hat_ = VectorField.FromData(
                FourierField, domain, velocity_trajectory[j], name="velocity_hat"
            )
            vel_ = vel_hat_.no_hat()
            vel_.set_time_step(j + time_step_0)
            vel_energy_ = vel_.energy()
            time_ = (vel_.get_time_step() / (n_steps - 1)) * end_time
            ts.append(time_)
            energy_t.append(vel_energy_)
            e_x_2d = vel_[0].hat().energy_2d(0)
            e_x_3d = vel_energy_ - e_x_2d
            energy_x_2d.append(e_x_2d)
            energy_x_3d.append(e_x_3d)
            amplitude_t.append(vel_[0].max() - vel_[0].min())
            vel_2d_x = vel_hat_[0].field_2d(0).no_hat()
            amplitude_x_2d_t.append(vel_2d_x.max() - vel_2d_x.min())
            vel_3d = vel_ - VectorField(
                [vel_2d_x, PhysicalField.Zeros(domain), PhysicalField.Zeros(domain)]
            )
            # amplitude_3d_t.append(vel_3d.max() - vel_3d.min())
            amplitude_3d_t.append(vel_3d[0].max() - vel_3d[0].min())
            # prod.append(nse.get_production(j))
            # diss.append(nse.get_dissipation(j))

        energy_t_arr = np.array(energy_t)
        energy_x_2d_arr = np.array(energy_x_2d)
        energy_x_3d_arr = np.array(energy_x_3d)
        print(max(energy_t_arr) / energy_t_arr[0])

        print("main post-processing loop")
        for i in range(n_steps):
            print("step", i + 1, "of", n_steps)
            # time = (i / (n_steps - 1)) * end_time
            vel_hat = VectorField.FromData(
                FourierField, domain, velocity_trajectory[i], name="velocity"
            )
            vel = vel_hat.no_hat()
            vel.set_time_step(i + time_step_0)

            vort = vel.curl()
            vel.set_name("velocity")
            vort.set_name("vorticity")
            time_step = vel.get_time_step()

            if i == 0:
                vel_shape = vel[0].get_data().shape
                max_inds = np.unravel_index(
                    vel[0].get_data().argmax(axis=None), vel_shape
                )
                Nx, _, Nz = vel_shape
                x_max = max_inds[0] / Nx * domain.grid[0][-1]
                z_max = max_inds[2] / Nz * domain.grid[2][-1]
            vel[0].plot_3d(2, z_max)
            vel[1].plot_3d(2, z_max)
            vel[2].plot_3d(2, z_max)
            vel[0].plot_3d(0, x_max)
            vel[1].plot_3d(0, x_max)
            vel[2].plot_3d(0, x_max)
            vel.plot_streamlines(2)
            vel[1].plot_isolines(2)
            vel.plot_isosurfaces()
            vel.plot_wavenumbers(1)
            vel.magnitude().plot_wavenumbers(1)

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            fig_amplitudes = figure.Figure()
            ax_amplitudes = fig_amplitudes.subplots(1, 1)
            assert type(ax_amplitudes) is Axes
            # fig_pd = figure.Figure()
            # ax_pd = fig_pd.subplots(1, 1)
            # assert type(ax_pd) is Axes

            # prod_arr = np.array(prod)
            # diss_arr = np.array(diss)
            ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
            ax.plot(
                ts[: i + 1],
                energy_t_arr[: i + 1] / energy_t_arr[0],
                "ko",
                label="G",
            )
            ax.set_xlabel("$t h / u_\\tau$")
            ax.set_ylabel("$G$")
            ax.plot(ts, energy_x_2d_arr / energy_t_arr[0], "b.")
            ax.plot(ts, energy_x_3d_arr / energy_t_arr[0], "g.")
            ax.plot(
                ts[: i + 1],
                energy_x_2d_arr[: i + 1] / energy_t_arr[0],
                "bo",
                label="$G_{x, 2d}$",
            )
            ax.plot(
                ts[: i + 1],
                energy_x_3d_arr[: i + 1] / energy_t_arr[0],
                "go",
                label="$G_{3d}$",
            )
            fig.legend()
            fig.savefig(
                Field.plotting_dir
                + "/plot_energy_t_"
                + "{:06}".format(time_step)
                + ".png"
            )
            # ax_2d_over_3d.set_yscale("log")
            ax_amplitudes.plot(ts, amplitude_t, "k.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_t[: i + 1],
                "ko",
                label="total perturbation x-velocity amplitude",
            )
            ax_amplitudes.plot(ts, amplitude_x_2d_t, "b.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_x_2d_t[: i + 1],
                "bo",
                label="streak amplitude (x-velocity)",
            )
            ax_amplitudes.plot(ts, amplitude_3d_t, "g.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_3d_t[: i + 1],
                "go",
                label="perturbation amplitude w/o streak (x-velocity)",
            )
            ax_amplitudes.set_xlabel("$t h / u_\\tau$")
            ax_amplitudes.set_ylabel("$A$")
            fig_amplitudes.legend()
            fig_amplitudes.savefig(
                Field.plotting_dir
                + "/plot_amplitudes_t_"
                + "{:06}".format(time_step)
                + ".png"
            )
            # ax_pd.plot(prod_arr, -diss_arr, "k.")
            # ax_pd.plot(
            #     np.array([0.0, max(-diss_arr)]),
            #     np.array([0.0, max(-diss_arr)]),
            #     color="0.8",
            #     linestyle="dashed",
            # )
            # ax_pd.plot(
            #     prod_arr[i],
            #     -diss_arr[i],
            #     "bo",
            # )
            # ax_pd.set_xlabel("$P$")
            # ax_pd.set_ylabel("$-D$")
            # fig_pd.savefig(
            #     Field.plotting_dir
            #     + "/plot_prod_diss_t_"
            #     + "{:06}".format(time_step)
            #     + ".png"
            # )


STORE_PREFIX = "/store/DAMTP/dsk34"
HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)
args = get_args_from_yaml_file(HOME_DIR_BASE + "/simulation_settings.yml")
assert len(sys.argv) > 1, "please provide a trajectory file to analyse"
assert (
    len(sys.argv) <= 2
), "there is no need to provide further arguments as these are inferred automatically from simulation_settings.yml"
post_process(
    sys.argv[1],
    args.get("end_time", 0.7),
    args.get("Lx_over_pi", 2.0),
    args.get("Lz_over_pi", 1.0),
    0,
)

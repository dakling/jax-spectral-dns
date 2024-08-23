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
import matplotlib
from matplotlib import figure
from matplotlib.axes import Axes
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, VectorField, PhysicalField, FourierField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

# matplotlib.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']


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
        energy_x_2d_1 = []
        energy_x_2d_2 = []
        amplitude_t = []
        amplitude_x_2d_t = []
        amplitude_x_2d_t_1 = []
        amplitude_x_2d_t_2 = []
        amplitude_z_2d_t = []
        amplitude_z_2d_t_1 = []
        amplitude_z_2d_t_2 = []
        amplitude_3d_t = []
        amplitudes_2d_kxs = []
        amplitudes_2d_kzs = []
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
            # e_x_2d = vel_[0].hat().energy_2d(0)
            e_x_2d = vel_.hat().energy_2d(0)
            e_x_3d = vel_energy_ - e_x_2d
            energy_x_2d.append(e_x_2d)
            energy_x_3d.append(e_x_3d)
            amplitude_t.append(vel_[0].max() - vel_[0].min())
            vel_2d_x = vel_hat_[0].field_2d(0).no_hat()
            vel_2d_x_1 = vel_hat_[0].field_2d(0, 1).no_hat()
            vel_2d_x_2 = vel_hat_[0].field_2d(0, 2).no_hat()
            vel_2d_z = vel_hat_[0].field_2d(2).no_hat()
            vel_2d_z_1 = vel_hat_[0].field_2d(2, 1).no_hat()
            vel_2d_z_2 = vel_hat_[0].field_2d(2, 2).no_hat()
            e_x_2d_1 = vel_2d_x_1.energy()
            e_x_2d_2 = vel_2d_x_2.energy()
            energy_x_2d_1.append(e_x_2d_1)
            energy_x_2d_2.append(e_x_2d_2)
            vel_2d_x.set_name("velocity_x_2d")
            vel_2d_x.plot_3d(0)
            vel_2d_x.plot_3d(2)
            amplitude_x_2d_t.append(vel_2d_x.max() - vel_2d_x.min())
            amplitude_x_2d_t_1.append(vel_2d_x_1.max() - vel_2d_x_1.min())
            amplitude_x_2d_t_2.append(vel_2d_x_2.max() - vel_2d_x_2.min())
            amplitude_z_2d_t.append(vel_2d_z.max() - vel_2d_z.min())
            amplitude_z_2d_t_1.append(vel_2d_z_1.max() - vel_2d_z_1.min())
            amplitude_z_2d_t_2.append(vel_2d_z_2.max() - vel_2d_z_2.min())
            Nx = domain.get_shape()[0]
            Nz = domain.get_shape()[2]
            amplitudes_2d_kx = []
            for kx in range((Nx - 1) // 2 + 1):
                vel_2d_kx = vel_hat_[0].field_2d(0, kx).no_hat()
                amplitudes_2d_kx.append(vel_2d_kx.max() - vel_2d_kx.min())
            amplitudes_2d_kz = []
            for kz in range((Nz - 1) // 2 + 1):
                vel_2d_kz = vel_hat_[0].field_2d(2, kz).no_hat()
                amplitudes_2d_kz.append(vel_2d_kz.max() - vel_2d_kz.min())
            amplitudes_2d_kxs.append(amplitudes_2d_kx)
            amplitudes_2d_kzs.append(amplitudes_2d_kz)
            fig = figure.Figure()
            ax = fig.subplots(2, 1)
            ax[0].plot(amplitudes_2d_kx, "k.")
            ax[1].plot(amplitudes_2d_kz, "k.")
            fig.savefig(
                "plots/plot_amplitudes_over_wns_t_" + "{:06}".format(j) + ".png"
            )
            vel_3d = vel_ - VectorField(
                [vel_2d_x, PhysicalField.Zeros(domain), PhysicalField.Zeros(domain)]
            )
            # amplitude_3d_t.append(vel_3d.max() - vel_3d.min())
            amplitude_3d_t.append(vel_3d[0].max() - vel_3d[0].min())
            # prod.append(nse.get_production(j))
            # diss.append(nse.get_dissipation(j))

        energy_t_arr = np.array(energy_t)
        energy_x_2d_arr = np.array(energy_x_2d)
        energy_x_2d_1_arr = np.array(energy_x_2d_1)
        energy_x_2d_2_arr = np.array(energy_x_2d_2)
        energy_x_3d_arr = np.array(energy_x_3d)
        print(max(energy_t_arr) / energy_t_arr[0])

        amplitudes_2d_kxs_arr = np.array(amplitudes_2d_kxs)
        amplitudes_2d_kzs_arr = np.array(amplitudes_2d_kzs)
        fig_kx = figure.Figure()
        ax_kx = fig_kx.subplots(1, 1)
        fig_kz = figure.Figure()
        ax_kz = fig_kz.subplots(1, 1)
        ax_kx.plot(amplitude_t, "k--", label="full")
        ax_kz.plot(amplitude_t, "k--", label="full")
        for kx in range((Nx - 1) // 2 + 1)[:10]:
            ax_kx.plot(amplitudes_2d_kxs_arr[:, kx], label="kx = " + str(kx))
        for kz in range((Nz - 1) // 2 + 1)[:10]:
            ax_kz.plot(amplitudes_2d_kzs_arr[:, kz], label="kz = " + str(kz))
        fig_kx.legend()
        fig_kx.savefig("plots/plot_amplitudes_kx" + ".png")
        fig_kz.legend()
        fig_kz.savefig("plots/plot_amplitudes_kz" + ".png")

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
            # ax.plot(ts, energy_x_2d_1_arr / energy_t_arr[0], "y.")
            # ax.plot(ts, energy_x_2d_2_arr / energy_t_arr[0], "m.")
            ax.plot(ts, energy_x_3d_arr / energy_t_arr[0], "g.")
            ax.plot(
                ts[: i + 1],
                energy_x_2d_arr[: i + 1] / energy_t_arr[0],
                "bo",
                label="$G_{kx = 0}$",
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
                label="streak (kx = 0) amplitude (x-velocity)",
            )
            ax_amplitudes.plot(ts, amplitude_x_2d_t_1, "y.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_x_2d_t_1[: i + 1],
                "yo",
                label="amplitude (kx = 1) (x-velocity)",
            )
            ax_amplitudes.plot(ts, amplitude_x_2d_t_2, "r.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_x_2d_t_2[: i + 1],
                "ro",
                label="amplitude (kx = 2)",
            )
            ax_amplitudes.plot(ts, amplitude_z_2d_t, "m.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_z_2d_t[: i + 1],
                "mo",
                label="kz = 0 amplitude",
            )
            ax_amplitudes.plot(ts, amplitude_z_2d_t_1, "c.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_z_2d_t_1[: i + 1],
                "co",
                label="amplitude (kz = 1)",
            )
            ax_amplitudes.plot(ts, amplitude_z_2d_t_2, "k.")
            ax_amplitudes.plot(
                ts[: i + 1],
                amplitude_z_2d_t_2[: i + 1],
                "ko",
                label="amplitude (kz = 2)",
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

            fig_kx = figure.Figure()
            ax_kx = fig_kx.subplots(1, 1)
            fig_kz = figure.Figure()
            ax_kz = fig_kz.subplots(1, 1)
            ax_kx.set_xlabel("$t h / u_\\tau$")
            ax_kx.set_ylabel("$\\tilde{u}_x$ ampl.")
            ax_kz.set_xlabel("$t h / u_\\tau$")
            ax_kz.set_ylabel("$\\tilde{u}_x$ ampl.")
            ax_kx.plot(amplitude_t, "k.")
            ax_kx.plot(amplitude_t[: i + 1], "ko", label="full")
            ax_kz.plot(amplitude_t, "k.")
            ax_kz.plot(amplitude_t[: i + 1], "ko", label="full")
            for kx in range((Nx - 1) // 2 + 1)[0:14:2]:
                dots = ax_kx.plot(amplitudes_2d_kxs_arr[:, kx], ".")
                ax_kx.plot(
                    amplitudes_2d_kxs_arr[: i + 1, kx],
                    "o",
                    color=dots[0].get_color(),
                    label="$k_x = " + str(kx) + "$",
                )
            for kz in range((Nz - 1) // 2 + 1)[0:10]:
                dots = ax_kz.plot(amplitudes_2d_kzs_arr[:, kz], ".")
                ax_kz.plot(
                    amplitudes_2d_kzs_arr[: i + 1, kz],
                    "o",
                    color=dots[0].get_color(),
                    label="$k_z = " + str(kz) + "$",
                )
            fig_kx.legend()
            fig_kx.savefig(
                "plots/plot_amplitudes_kx_t_" + "{:06}".format(time_step) + ".png"
            )
            fig_kz.legend()
            fig_kz.savefig(
                "plots/plot_amplitudes_kz_t_" + "{:06}".format(time_step) + ".png"
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

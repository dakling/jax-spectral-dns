#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation

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


def get_domain(shape):
    return PhysicalDomain.create(
        shape,
        (True, False, True),
        scale_factors=(2 * np.pi, 1.0, np.pi),
    )


def post_process(file: str, end_time: float, time_step_0: int = 0) -> None:
    Equation.initialize()
    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        n_steps = velocity_trajectory.shape[0]
        domain = get_domain(velocity_trajectory.shape[2:])

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
            e_x_3d = vel_.energy() - e_x_2d
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

            vel[0].plot_3d(2)
            vel[1].plot_3d(2)
            vel[2].plot_3d(2)
            vel[0].plot_3d(0)
            vel[1].plot_3d(0)
            vel[2].plot_3d(0)
            vel.plot_streamlines(2)
            vel[1].plot_isolines(2)
            vel.plot_isosurfaces()
            vel.plot_wavenumbers(1)
            vel.magnitude().plot_wavenumbers(1)

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            fig_2d_over_3d = figure.Figure()
            ax_2d_over_3d = fig_2d_over_3d.subplots(1, 1)
            assert type(ax_2d_over_3d) is Axes
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
                label="energy gain",
            )
            ax.set_xlabel("$t h / u_\\tau$")
            ax.set_ylabel("$G$")
            ax2 = ax.twinx()
            ax2.plot(ts, amplitude_t, "g.")
            ax2.plot(
                ts[: i + 1],
                amplitude_t[: i + 1],
                "go",
                label="total perturbation amplitude",
            )
            ax2.plot(ts, amplitude_x_2d_t, "b.")
            ax2.plot(
                ts[: i + 1], amplitude_x_2d_t[: i + 1], "bo", label="streak amplitude"
            )
            ax2.plot(ts, amplitude_3d_t, "y.")
            ax2.plot(
                ts[: i + 1],
                amplitude_3d_t[: i + 1],
                "yo",
                label="perturbation amplitude w/o streak",
            )
            ax2.set_ylabel("$A$")
            fig.legend()
            fig.savefig(
                Field.plotting_dir
                + "/plot_energy_t_"
                + "{:06}".format(time_step)
                + ".png"
            )
            ax_2d_over_3d.plot(ts, energy_x_2d_arr, "k.")
            ax_2d_over_3d.plot(ts, energy_x_3d_arr, "b.")
            ax_2d_over_3d.plot(
                ts[: i + 1], energy_x_2d_arr[: i + 1], "ko", label="$E_{x, 2d}$"
            )
            ax_2d_over_3d.plot(
                ts[: i + 1], energy_x_3d_arr[: i + 1], "bo", label="$E_{3d}$"
            )
            ax_2d_over_3d.set_xlabel("$t h / u_\\tau$")
            ax_2d_over_3d.set_ylabel("$E$")
            # ax_2d_over_3d.set_yscale("log")
            fig_2d_over_3d.legend()
            fig_2d_over_3d.savefig(
                Field.plotting_dir
                + "/plot_energy_2d_over_3d_t_"
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


post_process(sys.argv[1], float(sys.argv[2]), 0)

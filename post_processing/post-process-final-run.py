#!/usr/bin/env python3

import os

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
        physical_shape_passed=False,
    )


def post_process(file: str, end_time: float, time_step_0: int = 0) -> None:
    with h5py.File(file, "r") as f:
        velocity_trajectory = f["velocity_trajectory"]
        n_steps = velocity_trajectory.shape[0]
        domain = get_domain(velocity_trajectory.shape[2:])

        ts = []
        energy_t = []
        energy_x_2d = []
        energy_x_3d = []
        # prod = []
        # diss = []
        print("preparing")
        for j in range(n_steps):
            print("preparing, step", j, "of", n_steps)
            vel_hat_ = VectorField.FromData(
                FourierField, domain, velocity_trajectory[j], name="velocity"
            )
            vel_ = vel_hat_.no_hat()
            vel_.set_time_step(j + time_step_0)
            vel_energy_ = vel_.energy()
            time_ = (vel_.get_time_step() / (n_steps - 1)) * end_time
            ts.append(time_)
            energy_t.append(vel_energy_)
            e_x_2d = vel_[0].hat().energy_2d(0)
            e_x_3d = vel_[0].energy() - e_x_2d
            energy_x_2d.append(e_x_2d)
            energy_x_3d.append(e_x_3d)
            # prod.append(nse.get_production(j))
            # diss.append(nse.get_dissipation(j))

        energy_t_arr = np.array(energy_t)
        energy_x_2d_arr = np.array(energy_x_2d)
        energy_x_3d_arr = np.array(energy_x_3d)
        print(max(energy_t_arr))

        print("main post-processing loop")
        for i in range(n_steps):
            print("step", i, "of", n_steps)
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
            # vel.plot_isosurfaces()
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
                "bo",
            )
            ax.set_xlabel("$t h / u_\\tau$")
            ax.set_ylabel("$G$")
            fig.savefig(
                Field.plotting_dir
                + "/plot_energy_t_"
                + "{:06}".format(time_step)
                + ".png"
            )
            ax_2d_over_3d.plot(ts, energy_x_2d_arr, "k.")
            ax_2d_over_3d.plot(ts, energy_x_3d_arr, "b.")
            ax_2d_over_3d.plot(
                ts[: i + 1], energy_x_2d_arr[: i + 1], "ko", label="E_2d"
            )
            ax_2d_over_3d.plot(
                ts[: i + 1], energy_x_3d_arr[: i + 1], "bo", label="E_3d"
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


post_process("fields/velocity_trajectory", 35.0, 0)

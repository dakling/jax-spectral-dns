#!/usr/bin/env python3

import os
import sys

from matplotlib.pyplot import tight_layout

from jax_spectral_dns.equation import Equation
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

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
from jax_spectral_dns.navier_stokes_perturbation import (
    NavierStokesVelVortPerturbation,
    update_nonlinear_terms_high_performance_perturbation_skew_symmetric,
)

matplotlib.set_loglevel("error")

from PIL import Image

# matplotlib.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']
font_size = 24
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


STORE_PREFIX = "/store/DAMTP/dsk34"
HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)


def get_domain(shape, Lx_over_pi: float, Lz_over_pi: float):
    return PhysicalDomain.create(
        shape,
        (True, False, True),
        scale_factors=(Lx_over_pi * np.pi, 1.0, Lz_over_pi * np.pi),
    )


def get_vel_field_minimal_channel(domain: PhysicalDomain):

    cheb_coeffs = np.loadtxt(
        HOME_DIR_BASE + "/profiles/Re_tau_180_90_small_channel.csv", dtype=np.float64
    )

    Ny = domain.number_of_cells(1)
    U_mat = np.zeros((Ny, len(cheb_coeffs)))
    for i in range(Ny):
        for j in range(len(cheb_coeffs)):
            U_mat[i, j] = cheby(j, 0)(domain.grid[1][i])
    U_y_slice = U_mat @ cheb_coeffs
    nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    u_data = np.moveaxis(
        np.tile(np.tile(U_y_slice, reps=(nz, 1)), reps=(nx, 1, 1)), 1, 2
    )
    vel_base = VectorField(
        [
            PhysicalField(domain, jnp.asarray(u_data)),
            PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
        ]
    )
    vel_base.set_name("velocity_base")
    return vel_base


def post_process(
    file: str,
    end_time: float,
    Lx_over_pi: float,
    Lz_over_pi: float,
    Re_tau: float,
    time_step_0: int = 0,
) -> None:
    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        n_steps = velocity_trajectory.shape[0]
        domain = get_domain(velocity_trajectory.shape[2:], Lx_over_pi, Lz_over_pi)
        # fourier_domain = domain.hat()

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
        amplitudes_2d_vilda = []
        lambda_y_s = []
        lambda_z_s = []
        try:
            E_0 = get_vel_field_minimal_channel(domain).energy()
        except FileNotFoundError:
            E_0 = 1.0
        # prod = []
        # diss = []
        print("preparing")
        for j in range(n_steps):
            print("preparing, step", j + 1, "of", n_steps)
            vel_hat_ = VectorField.FromData(
                FourierField, domain, velocity_trajectory[j], name="velocity_hat"
            )
            vel_hat_.set_time_step(j + time_step_0)
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
            # amplitude_t.append(vel_[0].max() - vel_[0].min())
            amplitude_t.append(vel_[0].inf_norm())
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
            vel_2d_x.set_time_step(j)
            vel_2d_x.plot_3d(0, rotate=True)
            vel_2d_x.plot_3d(2)
            amplitude_x_2d_t.append(vel_2d_x.inf_norm())
            amplitude_x_2d_t_1.append(vel_2d_x_1.inf_norm())
            amplitude_x_2d_t_2.append(vel_2d_x_2.inf_norm())
            amplitude_z_2d_t.append(vel_2d_z.inf_norm())
            amplitude_z_2d_t_1.append(vel_2d_z_1.inf_norm())
            amplitude_z_2d_t_2.append(vel_2d_z_2.inf_norm())
            Nx = domain.get_shape()[0]
            Nz = domain.get_shape()[2]
            amplitudes_2d_kx = []
            for kx in range((Nx - 1) // 2 + 1):
                vel_2d_kx = vel_hat_[0].field_2d(0, kx).no_hat()
                # amplitudes_2d_kx.append(vel_2d_kx.max() - vel_2d_kx.min())
                amplitudes_2d_kx.append(vel_2d_kx.inf_norm())
            amplitudes_2d_kz = []
            for kz in range((Nz - 1) // 2 + 1):
                vel_2d_kz = vel_hat_[0].field_2d(2, kz).no_hat()
                # amplitudes_2d_kz.append(vel_2d_kz.max() - vel_2d_kz.min())
                amplitudes_2d_kz.append(vel_2d_kz.inf_norm())
            amplitudes_2d_kxs.append(amplitudes_2d_kx)
            amplitudes_2d_kzs.append(amplitudes_2d_kz)

            kx_max = np.argmax(amplitudes_2d_kx)
            kz_max = np.argmax(amplitudes_2d_kz)
            vel_0_hat_2d = vel_hat_.field_2d(0, kx_max).field_2d(2, kz_max)
            energy_of_highest_mode_ratio = abs(
                (vel_0_hat_2d.no_hat().energy() - vel_energy_) / vel_energy_
            )
            print("energy_of_highest_mode_ratio:", energy_of_highest_mode_ratio)

            energy_arr = np.vstack([np.array(ts), np.array(energy_t)])
            np.savetxt("plots/energy.txt", energy_arr.T)

            fig = figure.Figure()
            ax = fig.subplots(2, 1)
            ax[0].plot(amplitudes_2d_kx, "k.")
            ax[1].plot(amplitudes_2d_kz, "k.")
            fig.tight_layout()
            fig.savefig(
                "plots/plot_amplitudes_over_wns_t_" + "{:06}".format(j) + ".png",
                bbox_inches="tight",
            )
            vel_3d = vel_ - VectorField(
                [vel_2d_x, PhysicalField.Zeros(domain), PhysicalField.Zeros(domain)]
            )
            # amplitude_3d_t.append(vel_3d.max() - vel_3d.min())
            # amplitude_3d_t.append(vel_3d[0].max() - vel_3d[0].min())
            amplitude_3d_t.append(vel_3d[0].inf_norm())
            # prod.append(nse.get_production(j))
            # diss.append(nse.get_dissipation(j))
            amplitudes_2d_vilda.append(np.sqrt(vel_2d_x.energy() / E_0 * Re_tau))

            lambda_y, lambda_z = vel_hat_[0].get_streak_scales()
            print("lambda_y+:", lambda_y * Re_tau)
            print("lambda_z+:", lambda_z * Re_tau)
            lambda_y_s.append(lambda_y)
            lambda_z_s.append(lambda_z)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        ax.plot(ts, amplitudes_2d_vilda, "k.")
        ax.set_xlabel("$t u_\\tau / h$")
        ax.set_ylabel("$A \\sqrt{\\text{Re}_\\tau} $")
        fig.tight_layout()
        fig.savefig(
            "plots/plot_amplitudes_vilda.png",
            bbox_inches="tight",
        )
        energy_t_arr = np.array(energy_t)
        energy_x_2d_arr = np.array(energy_x_2d)
        energy_x_2d_1_arr = np.array(energy_x_2d_1)
        energy_x_2d_2_arr = np.array(energy_x_2d_2)
        energy_x_3d_arr = np.array(energy_x_3d)
        print(max(energy_t_arr) / energy_t_arr[0])

        amplitudes_2d_kxs_arr = np.array(amplitudes_2d_kxs)
        amplitudes_2d_kzs_arr = np.array(amplitudes_2d_kzs)
        # fig_kx_pub = figure.Figure()
        # ax_kx_pub = fig_kx_pub.subplots(1, 1)
        fig_k_size = (8, 6)
        fig_kx = figure.Figure(figsize=fig_k_size)
        ax_kx = fig_kx.subplots(1, 1)
        fig_kz = figure.Figure(figsize=fig_k_size)
        ax_kz = fig_kz.subplots(1, 1)
        ax_kx.set_yscale("log")
        ax_kz.set_yscale("log")
        ax_kx.set_xlabel("$t u_\\tau / h$")
        ax_kz.set_xlabel("$t u_\\tau / h$")
        ax_kx.set_ylabel("$|u|_\\text{inf}$")
        ax_kz.set_ylabel("$|u|_\\text{inf}$")
        # ax_kx_pub.plot(amplitude_t, "k--", label="full")
        ax_kx.plot(ts, amplitude_t, "k--", label="full")
        ax_kz.plot(ts, amplitude_t, "k--", label="full")
        for kx in range((Nx - 1) // 2 + 1)[:10]:
            kx_ = int(kx * 2 / Lx_over_pi)
            ax_kx.plot(ts, amplitudes_2d_kxs_arr[:, kx], label="kx = " + str(kx_))
            # ax_kx_pub.plot(amplitudes_2d_kxs_arr[:, kx], "-" label="kx = " + str(kx_))
        for kz in range((Nz - 1) // 2 + 1)[:10]:
            kz_ = int(kz * 2 / Lz_over_pi)
            ax_kz.plot(ts, amplitudes_2d_kzs_arr[:, kz], label="kz = " + str(kz_))
        # fig_kx.legend()
        ax_kx.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_kx.tight_layout()
        fig_kx.savefig("plots/plot_amplitudes_kx" + ".png", bbox_inches="tight")
        # fig_kz.legend()
        ax_kz.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_kz.tight_layout()
        fig_kz.savefig("plots/plot_amplitudes_kz" + ".png", bbox_inches="tight")

        fig_lambdas = figure.Figure()
        ax_lambdas = fig_lambdas.subplots(1, 1)
        ax_lambdas2 = ax_lambdas.twinx()
        ax_lambdas.plot(ts, Re_tau * np.array(lambda_y_s), "ko", label="$\\lambda_y^+$")
        ax_lambdas2.plot(
            ts, Re_tau * np.array(lambda_z_s), "bo", label="$\\lambda_z^+$"
        )
        ax_lambdas.set_xlabel("$t$")
        ax_lambdas.set_ylabel("$\\lambda_y^+$")
        ax_lambdas2.set_ylabel("$\\lambda_z^+$", color="blue")
        ax_lambdas2.tick_params(axis="y", labelcolor="blue")
        ax_lambdas.set_ylim(bottom=0)
        ax_lambdas2.set_ylim(bottom=0)
        # fig_lambdas.legend()
        fig_lambdas.tight_layout()
        fig_lambdas.savefig("plots/plot_lambdas" + ".png", bbox_inches="tight")

        fig_lambda_z = figure.Figure()
        ax_lambda_z = fig_lambda_z.subplots(1, 1)
        ax_lambda_z.plot(
            ts, Re_tau * np.array(lambda_z_s), "bo", label="$\\lambda_z^+$"
        )
        ax_lambda_z.plot(
            ts, [100.0 for _ in ts], "k-", label="$\\lambda_z^+_\\text{mean}$"
        )
        ax_lambda_z.plot(
            ts,
            [60.0 for _ in ts],
            "k--",
            label="$\\lambda_z^+_\\text{mean} - \\lambda_z^+_\\text{std}$",
        )
        ax_lambda_z.plot(
            ts,
            [140.0 for _ in ts],
            "k--",
            label="$\\lambda_z^+_\\text{mean} + \\lambda_z^+_\\text{std}$",
        )
        ax_lambda_z.axvline(x=0.1, linestyle="--", color="k")
        ax_lambda_z.axvline(x=0.7, linestyle="--", color="k")
        ax_lambda_z.set_xlabel("$t$")
        ax_lambda_z.set_ylim(bottom=0)
        # fig_lambda_z.legend()
        fig_lambda_z.tight_layout()
        fig_lambda_z.savefig("plots/plot_lambda_z" + ".png", bbox_inches="tight")

        lambda_arr = np.vstack(
            [np.array(ts), np.array(lambda_y_s), np.array(lambda_z_s)]
        )
        np.savetxt("plots/lambdas.txt", lambda_arr.T)

        try:
            vel_base = get_vel_field_minimal_channel(domain)
        except FileNotFoundError:
            vel_base = vel_ * 0.0

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
            vel_total_x = vel[0] + vel_base[0]
            vel_total_x.set_name("velocity_total_x")
            vel_total_x.plot_3d(2, z_max, name="$U_x$")
            # vel[0].plot_3d(2, z_max, name="$\\tilde{u}_x$", name_color="red")
            vel[0].plot_3d(2, z_max, name="$\\tilde{u}_x$")
            vel[1].plot_3d(2, z_max)
            vel[2].plot_3d(2, z_max)
            vel[0].plot_3d(
                # 0, x_max, rotate=True, name="$\\tilde{u}_x$", name_color="red"
                0,
                x_max,
                rotate=True,
                name="$\\tilde{u}_x$",
            )
            vel[0].plot_3d(0, x_max, rotate=True, name="$\\tilde{u}_x$", no_cb=True)
            vel[1].plot_3d(0, x_max, rotate=True)
            vel[2].plot_3d(0, x_max, rotate=True)
            vel.plot_streamlines(2)
            vel[1].plot_isolines(2)
            # vel[0].plot_isosurfaces(name="$u_x$", name_color="red")
            vel[0].plot_isosurfaces(name="$u_x$")
            vel[1].plot_isosurfaces()
            vel[2].plot_isosurfaces()

            # # pressure
            # pressure_poisson_source = PhysicalField.Zeros(domain)
            # for k in range(3):
            #     for j in range(3):
            #         pressure_poisson_source += -(vel[k] * vel[j]).diff(j).diff(k)
            #         pressure_poisson_source += -(vel_base[k] * vel[j]).diff(j).diff(k)
            #         pressure_poisson_source += -(vel[k] * vel_base[j]).diff(j).diff(k)
            # pressure_poisson_source.set_name("pressure_poisson_source")
            # pressure_poisson_source.plot_3d(0)
            # pressure_poisson_source.plot_3d(2)
            # pressure = pressure_poisson_source.hat().solve_poisson().no_hat()
            # # filter_field = PhysicalField.FromFunc(domain, lambda X: jnp.exp(1.0)**(- 20.0 * X[1]**10) + 0.0 * X[2])
            # # filter_field = PhysicalField.FromFunc(
            # #     domain, lambda X: jnp.exp(-((1.03 * X[1]) ** 40)) + 0.0 * X[2]
            # # )  # make sure that we are not messing with the boundary conditions
            # # pressure *= filter_field
            # pressure.update_boundary_conditions()
            # pressure.set_name("pressure")
            # pressure.set_time_step(vel.get_time_step())
            # if i == 0:
            #     vel_shape = vel[0].get_data().shape
            #     max_inds = np.unravel_index(
            #         pressure.get_data().argmax(axis=None), vel_shape
            #     )
            #     Nx, _, Nz = vel_shape
            #     x_max_pres = max_inds[0] / Nx * domain.grid[0][-1]
            #     z_max_pres = max_inds[2] / Nz * domain.grid[2][-1]
            # pressure.plot_3d(0, x_max_pres, rotate=True)
            # pressure.plot_3d(2, z_max_pres)
            # dp_dy = pressure.diff(1)
            # dp_dy.update_boundary_conditions()
            # dp_dy.set_name("dp_dy")
            # dp_dy.set_time_step(vel.get_time_step())
            # if i == 0:
            #     vel_shape = vel[0].get_data().shape
            #     max_inds = np.unravel_index(
            #         dp_dy.get_data().argmax(axis=None), vel_shape
            #     )
            #     Nx, _, Nz = vel_shape
            #     x_max_dpdy = max_inds[0] / Nx * domain.grid[0][-1]
            #     z_max_dpdy = max_inds[2] / Nz * domain.grid[2][-1]
            # dp_dy.plot_3d(0, x_max_dpdy, rotate=True)
            # dp_dy.plot_3d(2, z_max_dpdy)

            q_crit = vel.get_q_criterion()
            q_crit.set_time_step(vel.get_time_step())
            q_crit.set_name("q_criterion")
            q_crit.plot_3d(2, z_max)
            q_crit.plot_3d(0, x_max)
            vel.plot_q_criterion_isosurfaces(iso_vals=[0.05, 0.1, 0.5])
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
                label="$G$",
            )
            ax.set_xlabel("$t u_\\tau / h$")
            ax.set_ylabel("$G$")
            ax.plot(ts, energy_x_2d_arr / energy_t_arr[0], "b.")
            # ax.plot(ts, energy_x_2d_arr / energy_x_2d_arr[0], "b.")
            # ax.plot(ts, energy_x_2d_1_arr / energy_t_arr[0], "y.")
            # ax.plot(ts, energy_x_2d_2_arr / energy_t_arr[0], "m.")
            ax.plot(ts, energy_x_3d_arr / energy_t_arr[0], "g.")
            # ax.plot(ts, energy_x_3d_arr / energy_x_3d_arr[0], "g.")
            ax.plot(
                ts[: i + 1],
                energy_x_2d_arr[: i + 1] / energy_t_arr[0],
                # energy_x_2d_arr[: i + 1] / energy_x_2d_arr[0],
                "bo",
                label="$G_{k_x = 0}$",
            )
            ax.plot(
                ts[: i + 1],
                energy_x_3d_arr[: i + 1] / energy_t_arr[0],
                # energy_x_3d_arr[: i + 1] / energy_x_3d_arr[0],
                "go",
                label="$G_{3d}$",
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
            ax.set_box_aspect(1)
            fig.savefig(
                Field.plotting_dir
                + "/plot_energy_t_"
                + "{:06}".format(time_step)
                + ".png",
                bbox_inches="tight",
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
                label="streak (k_x = 0) amplitude (x-velocity)",
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
            ax_amplitudes.set_xlabel("$t u_\\tau / h$")
            ax_amplitudes.set_ylabel("$A$")
            fig_amplitudes.legend()
            fig_amplitudes.tight_layout()
            fig_amplitudes.savefig(
                Field.plotting_dir
                + "/plot_amplitudes_t_"
                + "{:06}".format(time_step)
                + ".png",
                bbox_inches="tight",
            )

            fig_kx = figure.Figure()
            ax_kx = fig_kx.subplots(1, 1)
            fig_kz = figure.Figure()
            ax_kz = fig_kz.subplots(1, 1)
            ax_kx.set_xlabel("$t u_\\tau / h$")
            ax_kx.set_ylabel(
                # "$\\textcolor{red}{\\tilde{u}_{x_\\text{max}}} - \\textcolor{red}{\\tilde{u}_{x_\\text{min}}}$"
                # "$\\tilde{u}_{x_\\text{max}} - \\tilde{u}_{x_\\text{min}}$"
                # "$\\tilde{u}_{x_\\text{maax}} - \\tilde{u}_{x_\\text{min}}$"
                "$|\\tilde{u}_{x}|_\\text{inf}$",
            )
            # ax_kx.set_ylabel("${\\tilde{u}_x}$ amplitude")
            # ax_kx.yaxis.label.set_color("red")
            ax_kz.set_xlabel("$t u_\\tau / h$")
            # ax_kz.set_ylabel("$\\textcolor{red}{\\tilde{u}_x}$ amplitude")
            ax_kz.set_ylabel(
                # "$\\textcolor{red}{\\tilde{u}_{x_\\text{max}}} - \\textcolor{red}{\\tilde{u}_{x_\\text{min}}}$"
                # "$\\tilde{u}_{x_\\text{max}} - \\tilde{u}_{x_\\text{min}}$"
                "$|\\tilde{u}_{x}|_\\text{inf}$",
            )
            # ax_kz.set_ylabel("${\\tilde{u}_x}$ amplitude")
            # ax_kz.yaxis.label.set_color("red")
            ax_kx.plot(ts, amplitude_t, "k.")
            ax_kx.plot(ts[: i + 1], amplitude_t[: i + 1], "ko", label="full")
            ax_kz.plot(ts, amplitude_t, "k.")
            ax_kz.plot(ts[: i + 1], amplitude_t[: i + 1], "ko", label="full")
            # for kx in range((Nx - 1) // 2 + 1)[0:14:2]:
            for kx in range((Nx - 1) // 2 + 1)[0:10]:
                dots = ax_kx.plot(ts, amplitudes_2d_kxs_arr[:, kx], ".")
                ax_kx.plot(
                    ts[: i + 1],
                    amplitudes_2d_kxs_arr[: i + 1, kx],
                    "o",
                    color=dots[0].get_color(),
                    label="$k_x = " + str(kx) + "$",
                )
            for kz in range((Nz - 1) // 2 + 1)[0:10]:
                dots = ax_kz.plot(ts, amplitudes_2d_kzs_arr[:, kz], ".")
                ax_kz.plot(
                    ts[: i + 1],
                    amplitudes_2d_kzs_arr[: i + 1, kz],
                    "o",
                    color=dots[0].get_color(),
                    label="$k_z = " + str(kz) + "$",
                )
            ax_kx.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
            ax_kx.set_box_aspect(1)
            fname_kx = "plots/plot_amplitudes_kx_t_" + "{:06}".format(time_step)
            try:
                fig_kx.savefig(
                    fname_kx + ".ps",
                    # fname_kx + ".png",
                    bbox_inches="tight",
                )
                psimage = Image.open(fname_kx + ".ps")
                psimage.load(scale=10, transparency=True)
                psimage.save(fname_kx + ".png", optimize=True)
                image = Image.open(fname_kx + ".png")
                imageBox = image.getbbox()
                cropped = image.crop(imageBox)
                cropped.save(fname_kx + ".png")
            except Exception:
                fig_kx.savefig(
                    # fname_kx + ".ps",
                    fname_kx + ".png",
                    bbox_inches="tight",
                )

            ax_kz.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
            ax_kz.set_box_aspect(1)
            fname_kz = "plots/plot_amplitudes_kz_t_" + "{:06}".format(time_step)
            try:
                fig_kz.savefig(
                    fname_kz + ".ps",
                    # fname_kz + ".png",
                    bbox_inches="tight",
                )
                psimage = Image.open(fname_kz + ".ps")
                psimage.load(scale=10, transparency=True)
                psimage.save(fname_kz + ".png", optimize=True)
                image = Image.open(fname_kz + ".png")
                imageBox = image.getbbox()
                cropped = image.crop(imageBox)
                cropped.save(fname_kz + ".png")
            except Exception:
                fig_kz.savefig(
                    # fname_kz + ".ps",
                    fname_kz + ".png",
                    bbox_inches="tight",
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

            # if i in [0, n_steps // 2, n_steps]:
            if True:
                fig_kx = figure.Figure()
                ax_kx = fig_kx.subplots(1, 1)
                ax_kx_2 = ax_kx.twinx()
                fig_kz = figure.Figure()
                ax_kz = fig_kz.subplots(1, 1)
                ax_kz_2 = ax_kz.twinx()
                ax_kx.set_xscale("log")
                ax_kz.set_xscale("log")
                ax_kx_2.set_xscale("log")
                ax_kz_2.set_xscale("log")
                ax_kx.set_xlabel("$k_x$")
                ax_kz.set_xlabel("$k_z$")
                ax_kx.set_ylabel("$E$")
                ax_kz.set_ylabel("$E$")
                ax_kx_2.set_ylabel("$A$")
                ax_kz_2.set_ylabel("$A$")
                kxs = vel_hat.get_fourier_domain().grid[0]
                kzs = vel_hat.get_fourier_domain().grid[2]
                energy_kx = []
                energy_kz = []
                energy_x_kx = []
                energy_x_kz = []
                amp_kx = []
                amp_kz = []

                for kx in range((len(kxs) - 1) // 2):
                    vel_2d_kx = vel_hat.field_2d(0, kx).no_hat()
                    energy_kx.append(vel_2d_kx.energy())
                    energy_x_kx.append(vel_2d_kx[0].energy())
                    # amp_kx.append(vel_2d_kx[0].max() - vel_2d_kx[0].min())
                    amp_kx.append(vel_2d_kx[0].inf_norm())

                for kz in range((len(kzs) - 1) // 2):
                    vel_2d_kz = vel_hat.field_2d(2, kz).no_hat()
                    energy_kz.append(vel_2d_kz.energy())
                    energy_x_kz.append(vel_2d_kz[0].energy())
                    # amp_kz.append(vel_2d_kz[0].max() - vel_2d_kz[0].min())
                    amp_kz.append(vel_2d_kz[0].inf_norm())

                ax_kx.plot(energy_kx, "ko")
                ax_kz.plot(energy_kz, "ko")
                ax_kx.plot(energy_x_kx, "bo")
                ax_kz.plot(energy_x_kz, "bo")
                ax_kx_2.plot(amp_kx, "ro")
                ax_kz_2.plot(amp_kz, "ro")

                fig_kx.savefig(
                    "plots/plot_energy_spectrum_kx_t_" + "{:06}".format(i) + ".png",
                    bbox_inches="tight",
                )
                fig_kz.savefig(
                    "plots/plot_energy_spectrum_kz_t_" + "{:06}".format(i) + ".png",
                    bbox_inches="tight",
                )


def post_process_pub(
    file: str,
    Lx_over_pi: float,
    Lz_over_pi: float,
) -> None:

    n_snapshots = 3
    fig_pub_x_plane = figure.Figure(layout="tight", figsize=(15, 15))
    ax_pub_x_plane = fig_pub_x_plane.subplots(1, n_snapshots)
    fig_pub_z_plane = figure.Figure(layout="tight", figsize=(15, 15))
    ax_pub_z_plane = fig_pub_x_plane.subplots(1, n_snapshots)

    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        domain = get_domain(velocity_trajectory.shape[2:], Lx_over_pi, Lz_over_pi)
        n_fields = len(velocity_trajectory)
        n = 0
        for j in range(n_snapshots):
            i = (n_fields - 1) * j // (n_snapshots - 1)
            vel_hat = VectorField.FromData(
                FourierField, domain, velocity_trajectory[i], name="velocity_hat"
            )
            vel = vel_hat.no_hat()
            vel.set_time_step(j)
            vel.set_name("vel_pub")
            vel[0].plot_3d_single(
                0, name="$\\tilde{u}_x$", ax=ax_pub_x_plane[n], fig=fig_pub_x_plane
            )
            vel[0].plot_3d_single(
                2, name="$\\tilde{u}_x$", ax=ax_pub_z_plane[n], fig=fig_pub_z_plane
            )
            vel[0].plot_3d_single(0, name="$\\tilde{u}_x$")
            vel[0].plot_3d_single(2, name="$\\tilde{u}_x$")
            n += 1
    fig_pub_x_plane.savefig(
        "plots/vel_pub_x_plane.png",
        bbox_inches="tight",
    )
    fig_pub_z_plane.savefig(
        "plots/vel_pub_z_plane.png",
        bbox_inches="tight",
    )


args = get_args_from_yaml_file(HOME_DIR_BASE + "/simulation_settings.yml")
# args = {}
assert len(sys.argv) > 1, "please provide a trajectory file to analyse"
assert (
    len(sys.argv) <= 2
), "there is no need to provide further arguments as these are inferred automatically from simulation_settings.yml"

post_process_pub(
    sys.argv[1],
    args.get("Lx_over_pi", 1.0),
    args.get("Lz_over_pi", 1.0),
)

post_process(
    sys.argv[1],
    args.get("end_time", 0.7),
    args.get("Lx_over_pi", 1.0),
    args.get("Lz_over_pi", 1.0),
    args.get("Re_tau", 180.0),
    0,
)

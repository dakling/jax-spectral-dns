#!/usr/bin/env python3

import os
import sys

from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import glob

from typing import List, TypeVar
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
    vel_base.set_name("average_velocity_vilda")
    return vel_base


def post_process_averages() -> None:

    def average_y_symm(vel: VectorField[PhysicalField]) -> VectorField[PhysicalField]:
        y_axis = 0  # assuming vel is only a slice
        vel_flip = VectorField(
            [
                PhysicalField(
                    vel.get_physical_domain(), jnp.flip(v.get_data(), axis=y_axis)
                )
                for v in vel
            ]
        )
        out = 0.5 * (vel_flip + vel)
        out.set_name(vel.get_name() + "_y_symm")
        return out

    T = TypeVar("T", bound="Field")

    def avg_fields(fs: List[T]) -> T:
        out = fs[0] * 0.0
        for f in fs:
            out += f
        return out / len(fs)

    Lx_over_pi = 0.6
    Lz_over_pi = 0.3
    Nx, Ny, Nz = (48, 129, 60)
    domain = get_domain((Nx, Ny, Nz), Lx_over_pi, Lz_over_pi)

    slice_domain = PhysicalDomain.create(
        (domain.get_shape_aliasing()[1],),
        (False,),
        scale_factors=(1.0,),
        aliasing=1,
    )
    # avg_vels = []
    # for f in glob.glob("avg_vel_20*", root_dir="./fields"):
    #     avg_vels.append(VectorField.FromFile(domain, f, "average_velocity"))

    vel_base_turb = get_vel_field_minimal_channel(domain)
    vel_00_s = []
    vel_00_symm_s = []
    # time_step = 15138
    # for fl in sorted(
    #     glob.glob("trajectory_00_20*", root_dir="./fields"),
    #     key=lambda f: os.path.getmtime("fields/" + f),
    # ):
    #     with h5py.File("fields/" + fl, "r") as f:
    #         velocity_00_trajectory = f["trajectory_00"]
    #         n_steps = velocity_00_trajectory.shape[0]
    #         for i in range(n_steps):
    #             vel_00_s.append(
    #                 VectorField.FromData(
    #                     PhysicalField,
    #                     slice_domain,
    #                     velocity_00_trajectory[i],
    #                     name="velocity_00",
    #                 )
    #             )
    #             vel_00 = vel_00_s[-1]
    #             vel_00.set_time_step(time_step)
    #             vel_00.set_name("velocity_spatial_average")
    #             # vel_00[0].plot_center(
    #             #     0,
    #             #     PhysicalField(
    #             #         slice_domain, vel_base_turb[0][0, :, 0], name="vilda"
    #             #     ),
    #             # )
    #             vel_00_symm = average_y_symm(vel_00_s[-1])
    #             vel_00_symm.set_time_step(time_step)
    #             # vel_00_symm[0].plot_center(
    #             #     0,
    #             #     PhysicalField(
    #             #         slice_domain, vel_base_turb[0][0, :, 0], name="vilda"
    #             #     ),
    #             # )
    #             vel_00_symm_s.append(vel_00_symm)
    #             time_step += 1

    # print_verb("Taking the average of", len(vel_00_s), "snapshots (equally weighted!)")
    # avg = avg_fields(vel_00_symm_s)

    print_verb("Reusing previously calculated velocity average")
    avg_0 = PhysicalField.FromFile(
        slice_domain, "average_velocity", name="average_velocity_x"
    )
    avg = VectorField([avg_0, 0 * avg_0, 0 * avg_0])
    # nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
    # u_data = np.moveaxis(
    #     np.tile(
    #         np.tile(vel_base_turb_slice.get_data(), reps=(nz, 1)), reps=(nx, 1, 1)
    #     ),
    #     1,
    #     2,
    # )
    # avg = VectorField(
    #     [
    #         PhysicalField(domain, jnp.asarray(u_data)),
    #         PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
    #         PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
    #     ]
    # )
    avg.set_name("average_velocity")
    avg.set_time_step(0)
    # avg[0].save_to_file("average_velocity")
    for vel_00_symm in vel_00_symm_s:
        vel_00_symm[0].plot_center(0, avg[0])
    for vel_00 in vel_00_s:
        vel_00[0].plot_center(0, avg[0])
    mass_flux = []
    vel_cl = []
    n_bins = 9
    for f in vel_00_s:
        mass_flux.append(f[0].volume_integral())
        vel_cl.append(f[0][Ny // 2])
    fig = figure.Figure()
    ax = fig.subplots(2, 2)
    ax[0][0].plot(mass_flux)
    ax[1][0].plot(vel_cl)
    ax[0][1].hist(mass_flux, bins=n_bins, orientation="horizontal")
    ax[1][1].hist(vel_cl, bins=n_bins, orientation="horizontal")
    fig.savefig("plots/mass_flux_over_time.png")

    # vel_cl_hist, vel_cl_bin_edges = np.histogram(vel_cl, bins=n_bins)
    # mass_flux_hist, mass_flux_bin_edges = np.histogram(mass_flux, bins=n_bins)
    # profile_hist = [[] for _ in range(len(vel_cl_hist))]
    # profile_hist_mf = [[] for _ in range(len(mass_flux_hist))]
    # for i in range(len(vel_cl)):
    #     vel_cl_current = vel_cl[i]
    #     mass_flux_current = mass_flux[i]
    #     for j in range(len(vel_cl_bin_edges) - 1):
    #         if (
    #             vel_cl_current >= vel_cl_bin_edges[j]
    #             and vel_cl_current < vel_cl_bin_edges[j + 1]
    #         ):
    #             profile_hist[j].append(vel_00_symm_s[i])
    #             break
    #     for k in range(len(vel_cl_bin_edges) - 1):
    #         if (
    #             mass_flux_current >= mass_flux_bin_edges[k]
    #             and mass_flux_current < mass_flux_bin_edges[k + 1]
    #         ):
    #             profile_hist_mf[k].append(vel_00_symm_s[i])
    #             break
    # profile_hist_avg = []
    # profile_hist_avg_mf = []
    # for interval in profile_hist:
    #     profile_hist_avg.append(avg_fields(interval))
    # for interval in profile_hist_mf:
    #     profile_hist_avg_mf.append(avg_fields(interval))
    # fig_hist_profiles = figure.Figure(layout="tight", figsize=(10, 3 * n_bins))
    # ax_hist_profiles = fig_hist_profiles.subplots(len(profile_hist_avg), 2)

    # def get_dissipation(vel: "VectorField[PhysicalField]") -> float:
    #     return ((vel[0].diff(0)) ** 2).volume_integral()

    # def get_mass_flux(vel: "VectorField[PhysicalField]") -> float:
    #     return vel[0].volume_integral()

    # for i in range(len(profile_hist_avg)):
    #     profile_hist_avg[i].set_name("hist_bin_" + str(i))
    #     print("n:", len(profile_hist[i]))
    #     print("dissipation:", get_dissipation(profile_hist_avg[i]))
    #     print("mass flux:", get_mass_flux(profile_hist_avg[i]))
    #     profile_hist_avg[i].set_time_step(0)
    #     profile_hist_avg[i][0].save_to_file("vel_hist_bin_" + str(i))
    #     profile_hist_avg[i][0].plot_center(
    #         0, avg[0], fig=fig_hist_profiles, ax=ax_hist_profiles[i][0]
    #     )
    #     profile_hist_avg_mf[i][0].plot_center(
    #         0, avg[0], fig=fig_hist_profiles, ax=ax_hist_profiles[i][1]
    #     )
    # fig_hist_profiles.savefig("plots/vel_hist_avg_profiles.png")

    print_verb("calculating statistics")
    with h5py.File("fields/" + "trajectory", "r") as f:
        vel_pert_s = []

        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = np.moveaxis(
            np.tile(np.tile(avg[0].get_data(), reps=(nz, 1)), reps=(nx, 1, 1)),
            1,
            2,
        )
        avg_ = VectorField(
            [
                PhysicalField(domain, jnp.asarray(u_data)),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            ]
        )
        velocity_trajectory = f["trajectory"]
        n_steps = velocity_trajectory.shape[0]
        for i in range(n_steps):
            vel_pert = (
                VectorField.FromData(
                    FourierField, domain, velocity_trajectory[i], name="velocity"
                ).no_hat()
                - avg_
            )
            vel_pert.set_name("velocity_pert")

            vel_pert_s.append(vel_pert)
        uu = (
            avg_fields([vel_pert[0] ** 2 for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        vv = (
            avg_fields([vel_pert[1] ** 2 for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        ww = (
            avg_fields([vel_pert[2] ** 2 for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        uv = (
            avg_fields([vel_pert[0] * vel_pert[1] for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        uw = (
            avg_fields([vel_pert[0] * vel_pert[2] for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        vw = (
            avg_fields([vel_pert[1] * vel_pert[2] for vel_pert in vel_pert_s])
            .hat()
            .field_2d(2)
            .field_2d(0)
            .no_hat()
        )
        fig_stat = figure.Figure()
        ax_stat = fig_stat.subplots(1, 1)
        uu.plot_center(1, fig=fig_stat, ax=ax_stat)
        vv.plot_center(1, fig=fig_stat, ax=ax_stat)
        ww.plot_center(1, fig=fig_stat, ax=ax_stat)
        uv.plot_center(1, fig=fig_stat, ax=ax_stat)
        uw.plot_center(1, fig=fig_stat, ax=ax_stat)
        vw.plot_center(1, fig=fig_stat, ax=ax_stat)
        fig_stat.savefig("plots/reystresses.png")


post_process_averages()

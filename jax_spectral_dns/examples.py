#!/usr/bin/env python3
from __future__ import annotations

import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import matplotlib.figure as figure
from matplotlib.axes import Axes
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Optional, Union, cast, Tuple, List
import time

from jax_spectral_dns import navier_stokes_perturbation
from jax_spectral_dns.cheb import cheb
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import (
    FourierField,
    PhysicalField,
    FourierFieldSlice,
    VectorField,
)
from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.navier_stokes import (
    NavierStokesVelVort,
    solve_navier_stokes_laminar,
)
from jax_spectral_dns.navier_stokes_perturbation import (
    NavierStokesVelVortPerturbation,
    solve_navier_stokes_perturbation,
)
from jax_spectral_dns.linear_stability_calculation import (
    LinearStabilityCalculation,
)
from jax_spectral_dns.navier_stokes_perturbation_dual import (
    perform_step_navier_stokes_perturbation_dual,
)
from jax_spectral_dns.optimiser import (
    OptimiserFourier,
    OptimiserNonFourier,
    OptimiserPertAndBase,
)

if TYPE_CHECKING:
    from jax_spectral_dns._typing import pseudo_2d_perturbation_return_type
    from jax_spectral_dns._typing import (
        jsd_float,
        jnp_array,
        np_float_array,
        Vel_fn_type,
        np_jnp_array,
        parameter_type,
    )

NoneType = type(None)


def run_navier_stokes_turbulent_pseudo_2d() -> None:
    Re = 5000

    end_time = 5

    use_antialias = False
    # use_antialias = True # also works fine
    if use_antialias:
        aliasing = 3 / 2
        Nz = 4
    else:
        aliasing = 1
        Nz = 2
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=128,
        Ny=128,
        Nz=Nz,
        dt=5e-3,
        end_time=end_time,
        scale_factors=(2 * np.pi, 1.0, 2 * np.pi * 1e-3),
        aliasing=aliasing,
    )
    u_fn = (
        lambda X: 1
        - X[1] ** 2
        + 0.1
        * (
            jnp.sin(X[0])
            * (
                jnp.cos(jnp.pi / 2 * X[1]) * (-2 * X[1])
                - jnp.pi / 2 * jnp.sin(jnp.pi / 2 * X[1]) * (1 - X[1] ** 2)
            )
        )
        + 0 * X[2]
    )
    v_fn = (
        lambda X: 0.1 * (-jnp.cos(X[0]) * (1 - X[1] ** 2) * jnp.cos(jnp.pi / 2 * X[1]))
        + 0 * X[2]
    )

    w_fn = lambda X: 0 * X[2]

    vel_x = PhysicalField.FromFunc(nse.get_physical_domain(), u_fn, name="velocity_x")
    vel_y = PhysicalField.FromFunc(nse.get_physical_domain(), v_fn, name="velocity_y")
    vel_z = PhysicalField.FromFunc(nse.get_physical_domain(), w_fn, name="velocity_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")
    vel_hat = vel.hat()
    vel_hat.set_name("velocity_hat")
    nse.init_velocity(vel_hat)

    u_base_fn = lambda X: 1 - X[1] ** 2
    v_base_fn = lambda X: 0.0 * jnp.sin(X[2])
    w_base_fn = lambda X: 0.0 * jnp.sin(X[2])

    vel_base_x = PhysicalField.FromFunc(
        nse.get_physical_domain(), u_base_fn, name="velocity_base_x"
    )
    vel_base_y = PhysicalField.FromFunc(
        nse.get_physical_domain(), v_base_fn, name="velocity_base_y"
    )
    vel_base_z = PhysicalField.FromFunc(
        nse.get_physical_domain(), w_base_fn, name="velocity_base_z"
    )
    vel_base = VectorField([vel_base_x, vel_base_y, vel_base_z], name="velocity_base")

    ts = []
    energy_t = []

    nse.initialize()
    nse.set_before_time_step_fn(None)
    nse.set_after_time_step_fn(None)
    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()

    n_steps = nse.get_number_of_fields("velocity_hat")

    def post_process(nse: NavierStokesVelVort, i: int) -> None:
        time = (i / (n_steps - 1)) * end_time
        vel = nse.get_field("velocity_hat", i).no_hat()
        vel_pert = vel - vel_base
        vel_pert.set_name("velocity_pert")
        vel_pert.set_time_step(i)
        # vort_hat, _ = nse.get_vorticity_and_helicity()
        # vort = vort_hat.no_hat()
        # vort.set_time_step(i)
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[2].plot_3d(2)
        vel_pert[0].plot_3d(2)
        vel_pert[1].plot_3d(2)
        vel_pert[2].plot_3d(2)
        # vort[0].plot_3d(2)
        # vort[1].plot_3d(2)
        # vort[2].plot_3d(2)
        vel[0].plot_center(1)
        vel[1].plot_center(1)
        vel[2].plot_center(1)
        ts.append(time)
        energy = vel_pert.energy()
        energy_t.append(energy)
        print_verb(time, ",", energy)

    nse.set_post_process_fn(post_process)
    nse.post_process()

    energy_t_arr = np.array(energy_t)
    fig = figure.Figure()
    ax = fig.subplots(2, 1)
    assert type(ax) is np.ndarray
    try:
        dedalus_data = np.genfromtxt(
            "./energy-dedalus.txt",
            # "./energy_dedalus_highre.txt",
            delimiter=",",
        ).T
        ax[0].plot(dedalus_data[0], dedalus_data[1], label="dedalus")
    except FileNotFoundError:
        print_verb("No dedalus data to compare results with were found.")
    # try:
    #     dedalus_data_small_dt = np.genfromtxt(
    #         # "./energy-dedalus.txt",
    #         "./energy_dedalus_highre_small_dt.txt",
    #         delimiter=",",
    #     ).T
    #     ax[0].plot(
    #         dedalus_data_small_dt[0],
    #         dedalus_data_small_dt[1],
    #         "--",
    #         label="dedalus (small dt)",
    #     )
    # except FileNotFoundError:
    #     print_verb("No dedalus small dt data to compare results with were found.")
    ax[0].plot(ts, energy_t_arr, "--", label="jax")
    try:
        ax[1].plot(dedalus_data[0], dedalus_data[1] / dedalus_data[1][0])
    except Exception:
        print_verb("No dedalus data to compare results with were found.")
    # try:
    #     ax[1].plot(
    #         dedalus_data_small_dt[0],
    #         dedalus_data_small_dt[1] / dedalus_data_small_dt[1][0],
    #         "--",
    #     )
    # except Exception:
    #     print_verb("No dedalus small dt data to compare results with were found.")
    ax[1].plot(ts, energy_t_arr / energy_t_arr[0], "--")
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$E$")
    ax[1].set_ylabel("$E/E_0$")
    fig.legend()
    fig.savefig("plots/energy.png")


def run_navier_stokes_turbulent() -> None:
    Re = 3000

    end_time = 1
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=64,
        Ny=64,
        Nz=64,
        # Nx=92,
        # Ny=128,
        # Nz=92,
        # Nx=4,
        # Ny=12,
        # Nz=4,
        dt=1e-3,
        end_time=end_time,
        scale_factors=(2 * np.pi, 1.0, 2 * np.pi),
        aliasing=3 / 2,
    )

    u_fn = (
        lambda X: 1
        - X[1] ** 2
        + 0.1
        * (
            jnp.sin(X[0])
            * (
                jnp.cos(jnp.pi / 2 * X[1]) * (-2 * X[1])
                - jnp.pi / 2 * jnp.sin(jnp.pi / 2 * X[1]) * (1 - X[1] ** 2)
            )
        )
        * jnp.sin(X[2])
    )
    v_fn = (
        lambda X: 0.1
        * (-jnp.cos(X[0]) * (1 - X[1] ** 2) * jnp.cos(jnp.pi / 2 * X[1]))
        * jnp.sin(X[2])
    )
    w_fn = lambda X: 0.05 * jnp.cos(jnp.pi / 2 * X[1]) * jnp.sin(2 * X[0]) + 0 * X[2]

    vel_x = PhysicalField.FromFunc(nse.get_physical_domain(), u_fn, name="velocity_x")
    vel_y = PhysicalField.FromFunc(nse.get_physical_domain(), v_fn, name="velocity_y")
    vel_z = PhysicalField.FromFunc(nse.get_physical_domain(), w_fn, name="velocity_z")
    vel = VectorField([vel_x, vel_y, vel_z], name="velocity")
    vel_hat = vel.hat()
    vel_hat.set_name("velocity_hat")
    nse.init_velocity(vel_hat)

    u_base_fn = lambda X: 1 - X[1] ** 2
    v_base_fn = lambda X: 0.0 * jnp.sin(X[2])
    w_base_fn = lambda X: 0.0 * jnp.sin(X[2])

    vel_base_x = PhysicalField.FromFunc(
        nse.get_physical_domain(), u_base_fn, name="velocity_base_x"
    )
    vel_base_y = PhysicalField.FromFunc(
        nse.get_physical_domain(), v_base_fn, name="velocity_base_y"
    )
    vel_base_z = PhysicalField.FromFunc(
        nse.get_physical_domain(), w_base_fn, name="velocity_base_z"
    )
    vel_base = VectorField([vel_base_x, vel_base_y, vel_base_z], name="velocity_base")

    ts = []
    energy_t = []

    nse.initialize()
    nse.set_before_time_step_fn(None)
    nse.set_after_time_step_fn(None)
    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()

    n_steps = nse.get_number_of_fields("velocity_hat")

    def post_process(nse: NavierStokesVelVort, i: int) -> None:
        time = (i / (n_steps - 1)) * end_time
        vel = nse.get_field("velocity_hat", i).no_hat()
        vel_pert = vel - vel_base
        vel_pert.set_name("velocity_pert")
        vel_pert.set_time_step(i)
        # vort_hat, _ = nse.get_vorticity_and_helicity()
        # vort = vort_hat.no_hat()
        # vort.set_time_step(i)
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[2].plot_3d(2)
        vel_pert[0].plot_3d(2)
        vel_pert[1].plot_3d(2)
        vel_pert[2].plot_3d(2)
        # vort[0].plot_3d(2)
        # vort[1].plot_3d(2)
        # vort[2].plot_3d(2)
        vel[0].plot_center(1)
        vel[1].plot_center(1)
        vel[2].plot_center(1)
        ts.append(time)
        energy = vel_pert.energy()
        energy_t.append(energy)
        print_verb(time, ",", energy)

    nse.set_post_process_fn(post_process)
    nse.post_process()

    energy_t_arr = np.array(energy_t)
    fig = figure.Figure()
    ax = fig.subplots(2, 1)
    assert type(ax) is np.ndarray
    try:
        dedalus_data = np.genfromtxt(
            # "./energy-dedalus.txt",
            "./energy_dedalus_highre.txt",
            delimiter=",",
        ).T
        ax[0].plot(dedalus_data[0], dedalus_data[1], label="dedalus")
    except FileNotFoundError:
        print_verb("No dedalus data to compare results with were found.")
    try:
        dedalus_data_small_dt = np.genfromtxt(
            # "./energy-dedalus.txt",
            "./energy_dedalus_highre_small_dt.txt",
            delimiter=",",
        ).T
        ax[0].plot(
            dedalus_data_small_dt[0],
            dedalus_data_small_dt[1],
            "--",
            label="dedalus (small dt)",
        )
    except FileNotFoundError:
        print_verb("No dedalus small dt data to compare results with were found.")
    ax[0].plot(ts, energy_t_arr, "o", label="jax")
    try:
        ax[1].plot(dedalus_data[0], dedalus_data[1] / dedalus_data[1][0])
    except Exception:
        print_verb("No dedalus data to compare results with were found.")
    try:
        ax[1].plot(
            dedalus_data_small_dt[0],
            dedalus_data_small_dt[1] / dedalus_data_small_dt[1][0],
            "--",
        )
    except Exception:
        print_verb("No dedalus small dt data to compare results with were found.")
    ax[1].plot(ts, energy_t_arr / energy_t_arr[0], "o")
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$E$")
    ax[1].set_ylabel("$E/E_0$")
    fig.legend()
    fig.savefig("plots/energy.png")


def run_pseudo_2d() -> None:
    Ny = 64
    # Ny = 24
    # Re = 5772.22
    # Re = 6000
    Re = 5500
    alpha = 1.02056
    # alpha = 1.0

    Nx = 496
    Nz = 4
    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=Ny)

    end_time = 100
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(4 * (2 * jnp.pi / alpha), 1.0, 1.0),
    )

    u = lsc.velocity_field_single_mode(nse.get_physical_domain())
    vel_x_hat: VectorField[FourierField] = nse.get_initial_field("velocity_hat")

    eps = 5e-3
    nse.init_velocity(vel_x_hat + (u * eps).hat())

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.get_physical_domain())
    energy_over_time_fn: Callable[["jsd_float"], "jsd_float"] = (
        lambda t: eps**2 * energy_over_time_fn_raw(t, None)
    )
    energy_x_over_time_fn: Callable[["jsd_float"], "jsd_float"] = (
        lambda t: eps**2 * lsc.energy_over_time(nse.get_physical_domain())[0](t, 0)
    )
    energy_y_over_time_fn: Callable[["jsd_float"], "jsd_float"] = (
        lambda t: eps**2 * lsc.energy_over_time(nse.get_physical_domain())[0](t, 1)
    )
    print_verb("eigenvalue: ", ev)
    plot_interval = 10

    vel_pert_0_hat = nse.get_initial_field("velocity_hat")[1]
    vel_pert_0: PhysicalField = vel_pert_0_hat.no_hat()
    vel_pert_0.name = "veloctity_y_0"
    ts: List[float] = []
    energy_t: List[float] = []
    energy_x_t: List[float] = []
    energy_y_t: List[float] = []
    energy_t_ana: List[float] = []
    energy_x_t_ana: List[float] = []
    energy_y_t_ana: List[float] = []

    def before_time_step(nse: NavierStokesVelVort) -> None:
        i = nse.time_step
        if i % plot_interval == 0:
            # vel_hat = nse.get_field("velocity_hat", i)
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            vel_x_max = vel[0].max()
            print_verb("vel_x_max: ", vel_x_max)
            vel_x_fn_ana: "Vel_fn_type" = (
                lambda X: -vel_x_max * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            )
            vel_x_ana = PhysicalField.FromFunc(
                nse.get_physical_domain(), vel_x_fn_ana, name="vel_x_ana"
            )
            # vel_1_lap_a = nse.get_field("v_1_lap_hat_a", i).no_hat()
            # vel_1_lap_a.plot_3d()
            vel_pert = VectorField([vel[0] - vel_x_ana, vel[1], vel[2]])
            # vel_hat_old = nse.get_field("velocity_hat", max(0, i - 1))
            # vel_old = vel_hat_old.no_hat()
            # vel_x_max_old = vel_old[0].max()
            # vel_x_fn_ana_old = (
            #     lambda X: -vel_x_max_old * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            # )
            # vel_x_ana_old = PhysicalField.FromFunc(
            #     nse.get_physical_domain(), vel_x_fn_ana_old, name="vel_x_ana_old"
            # )
            # vel_pert_old = VectorField(
            #     [vel_old[0] - vel_x_ana_old, vel_old[1], vel_old[2]]
            # )
            vel_pert_energy: "jsd_float" = 0.0
            v_1_lap_p = nse.get_latest_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p_0 = nse.get_initial_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p.time_step = i
            v_1_lap_p.plot_3d(2)
            for j in range(2):
                vel[j].time_step = i
                vel_pert[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vel[j].plot_3d(2)
                # vel[j].plot_center(0)
                if j == 0:
                    vel[j].plot_center(1, vel_x_ana)
                elif j == 1:
                    vel[j].plot_center(1, vel_pert_0)
                # vel_hat[j].plot_3d()
                vel_pert[j].name = "velocity_perturbation_" + "xyz"[j]
                vel_pert[j].plot_3d()
                vel_pert[j].plot_3d(2)
                # vel_pert[j].plot_center(0)
                # vel_pert[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            print_verb(
                "analytical velocity perturbation energy: ",
                energy_over_time_fn(nse.time),
            )
            print_verb("velocity perturbation energy: ", vel_pert_energy)
            print_verb("velocity perturbation energy x: ", vel_pert[0].energy())
            print_verb(
                "analytical velocity perturbation energy x: ",
                energy_x_over_time_fn(nse.time),
            )
            print_verb("velocity perturbation energy y: ", vel_pert[1].energy())
            print_verb(
                "analytical velocity perturbation energy y: ",
                energy_y_over_time_fn(nse.time),
            )
            print_verb("velocity perturbation energy z: ", vel_pert[1].energy())
            # vel_pert_energy_old = vel_pert_old.energy()
            # print_verb(
            #     "velocity perturbation energy change: ",
            #     vel_pert_energy - vel_pert_energy_old,
            # )
            # print_verb(
            #     "velocity perturbation energy x change: ",
            #     vel_pert[0].energy() - vel_pert_old[0].energy(),
            # )
            # print_verb(
            #     "velocity perturbation energy y change: ",
            #     vel_pert[1].energy() - vel_pert_old[1].energy(),
            # )
            # print_verb(
            #     "velocity perturbation energy z change: ",
            #     vel_pert[2].energy() - vel_pert_old[2].energy(),
            # )
            ts.append(float(nse.time))
            energy_t.append(float(vel_pert_energy))
            energy_x_t.append(float(vel_pert[0].energy()))
            energy_y_t.append(float(vel_pert[1].energy()))
            energy_t_ana.append(float(energy_over_time_fn(nse.time)))
            energy_x_t_ana.append(float(energy_x_over_time_fn(nse.time)))
            energy_y_t_ana.append(float(energy_y_over_time_fn(nse.time)))
            # if i > plot_interval * 3:
            if True:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                ax.plot(ts, energy_t_ana)
                ax.plot(ts, energy_t, ".")
                fig.savefig("plots/energy_t.png")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                ax.plot(ts, energy_x_t_ana)
                ax.plot(ts, energy_x_t, ".")
                fig.savefig("plots/energy_x_t.png")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                assert type(ax) is Axes
                ax.plot(ts, energy_y_t_ana)
                ax.plot(ts, energy_y_t, ".")
                fig.savefig("plots/energy_y_t.png")
        # input("carry on?")

    nse.set_before_time_step_fn(before_time_step)
    nse.set_after_time_step_fn(None)

    nse.solve()


def run_dummy_velocity_field() -> None:
    Re = 1e5

    end_time = 50

    nse = solve_navier_stokes_laminar(
        # Re=Re,
        # Ny=90,
        # Nx=64,
        # end_time=end_time,
        # perturbation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, perturbation_factor=1
        # Re=Re, Ny=60, Nx=32, end_time=end_time, perturbation_factor=1
        Re=Re,
        Ny=96,
        Nx=64,
        end_time=end_time,
        perturbation_factor=0,
    )

    sc_x = 1.87
    nse.max_iter = 1e10
    vel_x_fn = (
        lambda X: 0.0 * X[0] * X[1] * X[2]
        + jnp.cos(X[0] * 2 * jnp.pi / sc_x)
        + jnp.cos(X[1] * 2 * jnp.pi / 1.0)
    )
    vel_y_fn = (
        # lambda X: 0.0 * X[0] * X[1] * X[2] + X[0] * X[2] * (1 - X[1] ** 2) ** 2
        lambda X: 0.0 * X[0] * X[1] * X[2]
        + jnp.cos(X[1] * 2 * jnp.pi / 1.0)
    )  # fulfills bcs but breaks conti
    vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2]
    vel_x = PhysicalField.FromFunc(
        nse.get_physical_domain(), vel_x_fn, name="velocity_x"
    )
    vel_y = PhysicalField.FromFunc(
        nse.get_physical_domain(), vel_y_fn, name="velocity_y"
    )
    vel_z = PhysicalField.FromFunc(
        nse.get_physical_domain(), vel_z_fn, name="velocity_z"
    )
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 1

    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = PhysicalField.FromFunc(
        nse.get_physical_domain(), vel_x_fn_ana, name="vel_x_ana"
    )

    def after_time_step(nse: NavierStokesVelVort) -> None:
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_latest_field("velocity_hat").no_hat()
            vort_hat, _ = nse.get_vorticity_and_helicity()
            vort = vort_hat.no_hat()
            vel_pert = VectorField([vel[0] - vel_x_ana, vel[1], vel[2]])
            vel[0].plot_3d()
            vel[1].plot_3d()
            vel[2].plot_3d()
            vort[0].plot_3d()
            vort[1].plot_3d()
            vort[2].plot_3d()
            vel[0].plot_center(0)
            vel[1].plot_center(0)
            vel[2].plot_center(0)
            vel[0].plot_center(1)
            vel[1].plot_center(1)
            vel[2].plot_center(1)
            vel_pert_energy: "jsd_float" = 0.0
            vel_pert_abs: "jsd_float" = 0.0
            for j in range(3):
                vel_pert_energy += vel_pert[j].energy()
                vel_pert_abs += abs(vel_pert[j])
            print_verb("velocity perturbation energy: ", vel_pert_energy)
            print_verb("velocity perturbation abs: ", vel_pert_abs)

    nse.set_after_time_step_fn(after_time_step)
    nse.solve()


def run_pseudo_2d_perturbation(
    Re: float = 3000.0,
    alpha: float = 1.02056,
    end_time: float = 1.0,
    dt: float = 1e-2,
    Nx: int = 4,
    Ny: int = 96,
    Nz: int = 2,
    eps: float = 1e-0,
    linearize: bool = True,
    plot: bool = True,
    save: bool = True,
    v0: Optional["jnp_array"] = None,
    aliasing: float = 1.0,
    dealias_nonperiodic: bool = False,
    rotated: bool = False,
    jit: bool = True,
) -> "pseudo_2d_perturbation_return_type":
    Re = float(Re)
    alpha = float(alpha)
    end_time = float(end_time)
    dt = float(dt)
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    # Nx = float(Nx)
    # Ny = float(Ny)
    # Nz = float(Nz)

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=96)

    if not rotated:
        nse = solve_navier_stokes_perturbation(
            Re=Re,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            dt=dt,
            end_time=end_time,
            perturbation_factor=0.0,
            scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-3),
            aliasing=aliasing,
            dealias_nonperiodic=dealias_nonperiodic,
            rotated=False,
        )
    else:
        nse = solve_navier_stokes_perturbation(
            Re=Re,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            dt=dt,
            end_time=end_time,
            perturbation_factor=0.0,
            scale_factors=(1e-3, 1.0, 1 * (2 * jnp.pi / alpha)),
            aliasing=aliasing,
            dealias_nonperiodic=dealias_nonperiodic,
            rotated=True,
        )

    nse.set_linearize(linearize)
    # nse.initialize()

    if type(v0) == NoneType:
        U = lsc.velocity_field_single_mode(nse.get_physical_domain(), save=save)
    else:
        assert v0 is not None
        # U = VectorField([Field(nse.get_physical_domain(), v0[i]) for i in range(3)]).normalize()
        U = VectorField(
            [PhysicalField(nse.get_physical_domain(), v0[i]) for i in range(3)]
        )
        print_verb(U[0].energy())

    if rotated:
        U_ = lsc.velocity_field_single_mode(
            PhysicalDomain.create(
                (Nz, Ny, Nx),
                (True, False, True),
                scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-3),
                aliasing=aliasing,
            ),
            save=save,
        )
        U = VectorField(
            [
                PhysicalField(
                    nse.get_physical_domain(),
                    jnp.moveaxis(jnp.moveaxis(U_[2].data, 0, 2), 0, 1),
                ),
                PhysicalField(
                    nse.get_physical_domain(),
                    jnp.moveaxis(jnp.moveaxis(U_[1].data, 0, 2), 0, 1),
                ),
                PhysicalField(
                    nse.get_physical_domain(),
                    jnp.moveaxis(jnp.moveaxis(U_[0].data, 0, 2), 0, 1),
                ),
            ]
        )

    U_hat = U.hat()
    nse.init_velocity(U_hat * eps)

    energy_over_time_fn, _ = lsc.energy_over_time(nse.get_physical_domain(), eps=eps)

    vel_pert_0 = nse.get_initial_field("velocity_hat").no_hat()[1]
    vel_pert_0.name = "veloctity_y_0"
    ts: List[float] = []
    energy_t: List[float] = []
    energy_x_t: List[float] = []
    energy_y_t: List[float] = []
    energy_t_ana: List[float] = []
    energy_x_t_ana: List[float] = []
    energy_y_t_ana: List[float] = []

    nse.set_before_time_step_fn(None)
    nse.set_after_time_step_fn(None)

    if jit:
        nse.activate_jit()
    else:
        nse.deactivate_jit()
    nse.write_intermediate_output = plot
    nse.solve()
    nse.deactivate_jit()

    n_steps = nse.get_number_of_fields("velocity_hat")
    for i in range(n_steps):
        time = (i / (n_steps - 1)) * end_time
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        vort = vel.curl()
        vel.set_time_step(i)
        vort.set_time_step(i)
        vel.set_name("velocity")
        vort.set_name("vorticity")
        if rotated:
            vel[2].plot_3d(0)
        else:
            vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel_pert_energy = vel.energy()
        ts.append((time))
        energy_t.append((vel_pert_energy))
        energy_x_t.append((vel[0].energy()))
        energy_y_t.append((vel[1].energy()))
        energy_t_ana.append((energy_over_time_fn(time, None)))
        energy_x_t_ana.append((energy_over_time_fn(time, 0)))
        energy_y_t_ana.append((energy_over_time_fn(time, 1)))

    vel_pert = nse.get_latest_field("velocity_hat").no_hat()
    # vel_pert_old = nse.get_field("velocity_hat", nse.time_step - 3).no_hat()
    vel_pert_energy = vel_pert.energy()
    # vel_pert_energy_old = vel_pert_old.energy()

    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    assert type(ax) is Axes
    ax.plot(ts, energy_t, ".")
    ax.plot(ts, energy_t_ana, "-")
    fig.savefig("plots/energy_t.png")

    return (
        energy_t,
        energy_x_t,
        energy_y_t,
        energy_t_ana,
        energy_x_t_ana,
        energy_y_t_ana,
        ts,
        vel_pert,
    )


def run_jimenez_1990(start_time: int = 0) -> None:
    start_time = int(start_time)
    Re = 5000
    alpha = 1

    Nx = 100
    Ny = 140
    Nz = 2
    end_time = 1000

    nse = solve_navier_stokes_perturbation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(2 * jnp.pi / alpha, 1.0, 0.1),
    )

    nse.set_linearize(False)

    if start_time == 0:
        lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=Ny)
        vel_pert = lsc.velocity_field_single_mode(nse.get_physical_domain())
        vort_pert = vel_pert.curl()
        # eps = 1e0 / jnp.sqrt(vort_pert.energy())
        eps = 1e-2 / jnp.sqrt(vel_pert.energy())
        nse.init_velocity((vel_pert * eps).hat())
    else:
        u = PhysicalField.FromFile(
            nse.get_physical_domain(),
            "velocity_perturbation_" + str(0) + "_t_" + str(start_time),
            name="u_hat",
        )
        v = PhysicalField.FromFile(
            nse.get_physical_domain(),
            "velocity_perturbation_" + str(1) + "_t_" + str(start_time),
            name="v_hat",
        )
        w = PhysicalField.FromFile(
            nse.get_physical_domain(),
            "velocity_perturbation_" + str(2) + "_t_" + str(start_time),
            name="w_hat",
        )
        nse.init_velocity(
            VectorField(
                [
                    u.hat(),
                    v.hat(),
                    w.hat(),
                ]
            ),
        )
        nse.time_step = start_time

    plot_interval = 50

    def before_time_step(nse: NavierStokesVelVortPerturbation) -> None:
        i = nse.time_step
        if i % plot_interval == 0:
            vel_pert = nse.get_latest_field("velocity_hat").no_hat()
            vel_base = nse.get_latest_field("velocity_base_hat").no_hat()
            vel = vel_base + vel_pert
            vort = vel.curl()
            vort.set_name("vorticity")
            vort.set_time_step(i)
            vel_moving_frame = vel.shift(jnp.array([-0.353, 0, 0]))
            vel_moving_frame.set_name("velocity_moving_frame")
            vel_moving_frame.set_time_step(i)
            vel_moving_frame.plot_streamlines(2)
            # remove old fields
            for j in range(3):
                for f in Path("./fields/").glob(
                    "velocity_perturbation_" + str(j) + "_t_" + str(i - 10)
                ):
                    if f.is_file():
                        f.unlink()
                vel_pert[j].save_to_file(
                    "velocity_perturbation_" + str(j) + "_t_" + str(i)
                )
                vort[j].plot_3d()
                vort[j].plot_3d(2)
                vort[j].plot_isolines(2)
                # vel_moving_frame[j].plot_3d(2)

    nse.set_before_time_step_fn(before_time_step)
    nse.solve()


def run_transient_growth_nonpert(
    Re: float = 3000.0,
    T: float = 15.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    end_time: Optional[float] = None,
    eps: float = 1e-3,
    Nx: int = 6,
    Ny: int = 80,
    Nz: int = 6,
    plot: bool = True,
) -> Tuple[float, float, List[float], List[float]]:

    # ensure that these variables are not strings as they might be passed as command line arguments
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    eps = float(eps)

    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)

    if end_time is None:
        end_time = T
    else:
        assert end_time is not None
        end_time = float(end_time)
    number_of_modes = 60

    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi),
        dt=1e-2,
        # aliasing=3 / 2,
        aliasing=1,
    )
    # nse.initialize()

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=50)

    U_pert = lsc.calculate_transient_growth_initial_condition(
        nse.get_physical_domain(),
        T,
        number_of_modes,
        recompute_full=True,
        save_final=True,
    )

    velocity_x_base = PhysicalField.FromFunc(
        nse.get_physical_domain(),
        lambda X: 1 * (1 - X[1] ** 2) + 0.0 * X[0] * X[2],
        name="velocity_x_base",
    )
    velocity_y_base = PhysicalField.FromFunc(
        nse.get_physical_domain(),
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_y_base",
    )
    velocity_z_base = PhysicalField.FromFunc(
        nse.get_physical_domain(),
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_z_base",
    )
    velocity_base = VectorField([velocity_x_base, velocity_y_base, velocity_z_base])
    velocity_base.set_name("velocity_base")

    eps_ = eps / jnp.sqrt(U_pert.energy())
    U_pert_norm = U_pert * eps_

    U = velocity_base + U_pert_norm

    U_hat = U.hat()

    nse.init_velocity(U_hat)

    e_max = lsc.calculate_transient_growth_max_energy(T, number_of_modes)

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    if plot and abs(Re - 3000) < 1e-3:
        rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    def post_process(nse: NavierStokesVelVort, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        vel_pert = vel - velocity_base
        time = (i / (n_steps - 1)) * end_time

        if plot:
            vort_pert = vel_pert.curl()
            vel_pert.set_time_step(i)
            vel_pert.set_name("velocity_pert")
            vort_pert.set_time_step(i)
            vort_pert.set_name("vorticity_pert")
            vel_pert[0].plot_3d(2)
            vel_pert[1].plot_3d(2)
            vort_pert[2].plot_3d(2)
            vel_pert.plot_streamlines(2)
            vel_pert[0].plot_isolines(2)

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ts_ = []
            energy_t_ = []
            for j in range(n_steps):
                time_ = (j / (n_steps - 1)) * end_time
                vel_hat_ = nse.get_field("velocity_hat", j)
                vel_ = vel_hat.no_hat()
                vel_energy_ = vel_.energy()
                ts_.append(time_)
                energy_t_.append(vel_energy_)

            energy_t_arr = np.array(energy_t_)
            ax.plot(ts_, energy_t_arr / energy_t_arr[0], "k.")
            ax.plot(
                ts_[: i + 1],
                energy_t_arr[: i + 1] / energy_t_arr[0],
                "bo",
                label="energy gain",
            )
            ax.set_xlabel("$t$")
            ax.set_ylabel("$G$")
            fig.legend()
            fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        vel_pert_energy = vel_pert.energy()
        ts.append(time)
        energy_t.append(vel_pert_energy)

    nse.set_before_time_step_fn(None)
    nse.set_after_time_step_fn(None)
    nse.set_post_process_fn(post_process)

    nse.activate_jit()
    nse.write_intermediate_output = plot
    nse.solve()
    nse.deactivate_jit()
    nse.post_process()

    # energy_t = np.array(energy_t)
    # if plot:
    #     fig = figure.Figure()
    #     ax = fig.subplots(1, 1)
    #     ax.plot(ts, energy_t / energy_t[0], ".", label="growth (DNS)")
    #     if abs(Re - 3000) < 1e-3:
    #         ax.plot(
    #             rh_93_data[0],
    #             rh_93_data[1],
    #             "--",
    #             label="growth (Reddy/Henningson 1993)",
    #         )
    #     fig.legend()
    #     fig.savefig("plots/energy_t.png")

    gain = energy_t[-1] / energy_t[0]
    print_verb("final energy gain:", gain)
    print_verb("expected final energy gain:", e_max)

    return (gain, e_max, ts, energy_t)


def run_transient_growth(
    Re: float = 3000.0,
    T: float = 15.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    end_time: Optional[float] = None,
    eps: float = 1e-5,
    Nx: int = 4,
    Ny: int = 50,
    Nz: int = 4,
    linearize: Union[bool, str] = True,
    plot: bool = True,
) -> Tuple[float, float, List[float], List[float]]:

    # ensure that these variables are not strings as they might be passed as command line arguments
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)
    if type(linearize) == str:
        linearize_ = linearize == "True"
    else:
        assert type(linearize) is bool
        linearize_ = linearize

    eps = float(eps)

    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)

    if end_time is None:
        end_time = T
    else:
        assert end_time is not None
        end_time = float(end_time)
    number_of_modes = 60

    nse = solve_navier_stokes_perturbation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 0.93),
        dt=1e-2,
        aliasing=3 / 2,
        # aliasing=1,
    )
    # nse.initialize()

    nse.set_linearize(linearize_)

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=50)

    U = lsc.calculate_transient_growth_initial_condition(
        nse.get_physical_domain(),
        T,
        number_of_modes,
        recompute_full=True,
        save_final=True,
    )

    U_hat = U.hat()
    eps_ = eps / jnp.sqrt(U.energy())

    nse.init_velocity(U_hat * eps_)

    e_max = lsc.calculate_transient_growth_max_energy(T, number_of_modes)

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    if plot and abs(Re - 3000) < 1e-3:
        rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        time = (i / (n_steps - 1)) * end_time

        if plot:
            vort = vel.curl()
            vel.set_time_step(i)
            vel.set_name("velocity")
            vort.set_time_step(i)
            vort.set_name("vorticity")
            vel[0].plot_3d(2)
            vel[1].plot_3d(2)
            vort[2].plot_3d(2)
            vel.plot_streamlines(2)
            vel[0].plot_isolines(2)

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ts_ = []
            energy_t_ = []
            for j in range(n_steps):
                time_ = (j / (n_steps - 1)) * end_time
                vel_hat_ = nse.get_field("velocity_hat", j)
                vel_ = vel_hat_.no_hat()
                vel_energy_ = vel_.energy()
                ts_.append(time_)
                energy_t_.append(vel_energy_)

            energy_t_arr = np.array(energy_t_)
            ax.plot(ts_, energy_t_arr / energy_t_arr[0], "k.")
            ax.plot(
                ts_[: i + 1],
                energy_t_arr[: i + 1] / energy_t_arr[0],
                "bo",
                label="energy gain",
            )
            ax.set_xlabel("$t$")
            ax.set_ylabel("$G$")
            fig.legend()
            fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        vel_energy = vel.energy()
        ts.append(time)
        energy_t.append(vel_energy)

    nse.set_before_time_step_fn(None)
    nse.set_after_time_step_fn(None)
    nse.set_post_process_fn(post_process)

    nse.activate_jit()
    nse.write_intermediate_output = plot
    nse.solve()
    nse.deactivate_jit()
    nse.post_process()

    energy_t_arr = np.array(energy_t)
    if plot:
        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ax.plot(ts, energy_t_arr / energy_t_arr[0], ".", label="growth (DNS)")
        if abs(Re - 3000) < 1e-3:
            ax.plot(
                rh_93_data[0],
                rh_93_data[1],
                "--",
                label="growth (Reddy/Henningson 1993)",
            )
        fig.legend()
        fig.savefig("plots/energy_t.png")

    gain = energy_t_arr[-1] / energy_t_arr[0]
    print_verb("final energy gain:", gain)
    print_verb("expected final energy gain:", e_max)

    return (gain, e_max, ts, energy_t)


def run_transient_growth_time_study(
    transient_growth_fn_: Union[
        str, Callable[[float], Tuple[float, float, List[float], List[float]]]
    ] = run_transient_growth
) -> None:

    if type(transient_growth_fn_) is str:
        transient_growth_fn = globals()[sys.argv[2]]
    else:
        transient_growth_fn = transient_growth_fn_
    Re = 3000

    rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    assert type(ax) is Axes
    ax.set_xlabel("T")
    ax.set_ylabel("G")
    ax.plot(
        rh_93_data[0],
        rh_93_data[1],
        "b--",
        label="max gain (Reddy/Henningson 1993)",
    )
    fig.legend()
    fig.savefig("plots/energy_t_intermediate.png")
    ts_list = []
    energy_t_list = []
    T_list = np.arange(5, 41, 5)
    for T in np.flip(T_list):
        print_verb(
            "running transient growth calculation for time horizon of "
            + str(T)
            + " time units"
        )
        _, _, ts, energy_t = transient_growth_fn(Re, T, 1, 0)

        ts_list.append(ts)
        energy_t_list.append(energy_t)
        ax.plot(ts, energy_t / energy_t[0], ".", label="gain (T = " + str(T) + ")")
        ax.plot(
            rh_93_data[0],
            rh_93_data[1],
            "b--",
        )
        fig.legend()
        fig.savefig("plots/energy_t_intermediate.png")

    ts_list.reverse()
    energy_t_list.reverse()

    # make a nice final figure
    fig_final = figure.Figure()
    ax_final = fig_final.subplots(1, 1)
    assert type(ax_final) is Axes
    ax_final.set_xlabel("T")
    ax_final.set_ylabel("G")
    ax_final.plot(
        rh_93_data[0],
        rh_93_data[1],
        "b--",
        label="max gain (Reddy/Henningson 1993)",
    )
    for i in range(len(T_list)):
        T = T_list[i]
        ts = ts_list[i]
        energy_t = energy_t_list[i]
        ax_final.plot(
            ts, energy_t / energy_t[0], ".", label="gain (T = " + str(T) + ")"
        )

    ax_final.plot(
        rh_93_data[0],
        rh_93_data[1],
        "b--",
    )
    fig_final.legend()
    fig_final.savefig("plots/energy_t_final.png")


def run_optimisation_transient_growth(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 8,
    Ny: int = 90,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    dt = 1e-2
    end_time = T
    number_of_modes = 20  # deliberately low value so that there is room for improvement
    # number_of_modes = 60
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)
    # aliasing = 3 / 2
    aliasing = 1

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    e_0 = 1e-6
    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0
    v0_0_hat = v0_0_norm.hat()

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    def run_case(U_hat: VectorField[FourierField], out: bool = False) -> "jsd_float":

        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserFourier(
        domain,
        domain,
        run_case,
        v0_0_hat,
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
        add_noise=False,
    )
    optimiser.optimise()


def run_optimisation_transient_growth_nonfourier(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 8,
    Ny: int = 90,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    dt = 1e-2
    end_time = T
    number_of_modes = 20  # deliberately low value so that there is room for improvement
    # number_of_modes = 60
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)
    aliasing = 3 / 2

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    e_0 = 1e-6
    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    def run_case(U: VectorField[PhysicalField], out: bool = False) -> "jsd_float":

        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserNonFourier(
        domain,
        domain,
        run_case,
        v0_0_norm,
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
        add_noise=False,
    )
    optimiser.optimise()


def run_optimisation_transient_growth_y_profile(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 8,
    Ny: int = 90,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = 4,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    dt = 1e-2

    Equation.initialize()
    end_time = T
    number_of_modes = 20  # deliberately low value so that there is room for improvement
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi)
    # aliasing = 3 / 2
    aliasing = 1

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    e_0 = 1e-6
    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0
    v0_0_hat = v0_0_norm.hat()

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[0].plot_center(1)
        vel[1].plot_center(1)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")

    def run_case(U_hat: VectorField[FourierField], out: bool = False) -> "jsd_float":
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    def run_input_to_params(vel_hat: "VectorField[FourierField]") -> "parameter_type":
        v0_1 = vel_hat[1].data[1, :, 0] * (1 + 0j)
        v0_0_00_hat = vel_hat[0].data[0, :, 0] * (1 + 0j)
        v0 = (v0_1, v0_0_00_hat)
        return v0

    def params_to_run_input(params: "parameter_type") -> VectorField[FourierField]:
        v1_yslice = params[0]
        v1_hat = domain.field_hat(lsc.y_slice_to_3d_field(domain, v1_yslice))
        v0_00 = params[1]
        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
            domain, None, v1_hat, v0_00, None, two_d=True
        )
        U_hat: VectorField[FourierField] = VectorField.FromData(
            FourierField, domain, U_hat_data
        )
        return U_hat

    optimiser = OptimiserFourier(
        domain,
        domain,
        run_case,
        v0_0_hat,
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        run_input_to_parameters_fn=run_input_to_params,
        parameters_to_run_input_fn=params_to_run_input,
        objective_fn_name="gain",
        add_noise=False,
    )
    optimiser.optimise()


def run_optimisation_transient_growth_nonlinear(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 48,
    Ny: int = 64,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
    e_0: float = 1e-3,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    e_0 = float(e_0)
    dt = 1e-2
    end_time = T
    number_of_modes = 60
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)
    aliasing = 3 / 2

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0
    v0_0_hat = v0_0_norm.hat()

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    def run_case(U_hat: VectorField[FourierField], out: bool = False) -> "jsd_float":

        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        nse.set_linearize(False)
        # nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserFourier(
        domain,
        domain,
        run_case,
        v0_0_hat,
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
    )
    optimiser.optimise()


def run_optimisation_transient_growth_nonlinear_3d(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 48,
    Ny: int = 64,
    Nz: int = 48,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
    e_0: float = 1e-3,
    init_file: Optional[str] = None,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    # alpha=2*jnp.pi/1.87
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    e_0 = float(e_0)
    dt = 2e-2
    end_time = T
    number_of_modes = 60
    # scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 1.0)
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 0.93)
    # scale_factors = (1.87, 1.0, 0.93)
    aliasing = 3 / 2

    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    if init_file is None:
        lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

        v0_0 = lsc.calculate_transient_growth_initial_condition(
            domain,
            T,
            number_of_modes,
            recompute_full=True,
            save_final=False,
        )
        v0_0_norm = v0_0.normalize_by_energy()
        v0_0_norm *= e_0
        v0_0_hat = v0_0_norm.hat()
    else:
        v0_0_hat = None

    run_input_initial = init_file or v0_0_hat
    assert run_input_initial is not None

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    def run_case(U_hat: VectorField[FourierField], out: bool = False) -> "jsd_float":

        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        nse.set_linearize(False)
        # nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserFourier(
        domain,
        domain,
        run_case,
        run_input_initial,
        minimise=False,
        force_2d=False,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
        # add_noise=False,
        add_noise=True,
        noise_amplitude=1e-3,
    )
    optimiser.optimise()


def run_optimisation_transient_growth_nonlinear_3d_nonfourier(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 48,
    Ny: int = 64,
    Nz: int = 48,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
    e_0: float = 1e-3,
    init_file: Optional[str] = None,
) -> None:
    Re = float(Re)
    T = float(T)
    # alpha=1.0
    alpha = 2 * jnp.pi / 1.87
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    e_0 = float(e_0)
    dt = 1e-2
    end_time = T
    number_of_modes = 60
    # scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 1.0)
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 0.93)
    # scale_factors = (1.87, 1.0, 0.93)
    aliasing = 3 / 2

    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    if init_file is None:
        lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

        v0_0 = lsc.calculate_transient_growth_initial_condition(
            domain,
            T,
            number_of_modes,
            recompute_full=True,
            save_final=False,
        )
        v0_0_norm = v0_0.normalize_by_energy()
        v0_0_norm *= e_0
    else:
        v0_0_norm = None

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    def run_case(U: VectorField[PhysicalField], out: bool = False) -> "jsd_float":
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re)
        nse.end_time = end_time

        nse.set_linearize(False)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserNonFourier(
        domain,
        domain,
        run_case,
        init_file or v0_0_norm,
        minimise=False,
        force_2d=False,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
    )
    optimiser.optimise()


def run_optimisation_transient_growth_mean_y_profile(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 8,
    Ny: int = 90,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    dt = 1e-2

    Equation.initialize()
    end_time = T
    number_of_modes = 20  # deliberately low value so that there is room for improvement
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi)
    aliasing = 3 / 2

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    lsc0 = LinearStabilityCalculation(Re=Re, alpha=0, beta=0, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    e_0 = 1e-3
    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0
    v0_0_hat = v0_0_norm.hat()

    u_max_over_u_tau = 1.0

    velocity_x_base = PhysicalField.FromFunc(
        domain,
        lambda X: u_max_over_u_tau * (1 - X[1] ** 2) + 0.0 * X[0] * X[2],
        name="velocity_x_base",
    )
    velocity_y_base = PhysicalField.FromFunc(
        domain,
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_y_base",
    )
    velocity_z_base = PhysicalField.FromFunc(
        domain,
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_z_base",
    )
    velocity_base = VectorField([velocity_x_base, velocity_y_base, velocity_z_base])
    velocity_base_hat = velocity_base.hat()
    velocity_base_hat.set_name("velocity_base_hat")

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        if i == 0:
            vel_base = nse.get_initial_field("velocity_base_hat").no_hat()
            vel_base.plot_3d(2)
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[0].plot_center(1)
        vel[1].plot_center(1)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")

    def run_case(
        inp: tuple[VectorField[FourierField], VectorField[FourierField]],
        out: bool = False,
    ) -> "jsd_float":
        U_hat, U_base_hat = inp
        U_base = U_base_hat.no_hat()
        U_base.normalize_by_max_value()
        U_base_hat = U_base.hat()
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(
            U_norm, dt=dt, Re=Re, velocity_base_hat=U_base_hat
        )
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.set_before_time_step_fn(None)
        nse.set_after_time_step_fn(None)
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = OptimiserPertAndBase(
        domain,
        domain,
        run_case,
        (v0_0_hat, velocity_base_hat),
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
    )
    optimiser.optimise()


def run_ld_2021_get_mean() -> None:
    Re = 3275
    Nx: int = 64
    Ny: int = 129
    Nz: int = 64
    max_cfl = 0.3
    end_time = 100

    Equation.initialize()

    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=1,
    )
    coarse_domain = PhysicalDomain.create(
        (16, Ny, 16),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=1,
    )
    dt = Equation.find_suitable_dt(domain, max_cfl, (1.0, 1e-5, 1e-5), end_time)
    print_verb("dt:", dt)

    def post_process(nse: NavierStokesVelVort, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        avg_vel = VectorField([PhysicalField.Zeros(domain) for _ in range(3)])
        if i == 0:
            for j in range(n_steps):
                vel_hat = nse.get_field("velocity_hat", j)
                vel = vel_hat.no_hat()
                avg_vel += vel / n_steps
            slice_domain = PhysicalDomain.create(
                (Ny,),
                (False,),
                scale_factors=(1.0,),
                aliasing=1,
            )
            avg_vel.set_time_step(0)
            avg_vel.set_name("average_velocity")
            avg_vel.save_to_file("avg_vel")
            avg_vel[0].plot_3d(2)
            avg_vel[1].plot_3d(2)
            avg_vel[2].plot_3d(2)
            avg_vel_x_slice = PhysicalField.Zeros(slice_domain)
            for i_x in range(Nx):
                for i_z in range(Nz):
                    avg_vel_x_slice += avg_vel[0][i_x, :, i_z] / (Nx * Nz)
            avg_vel_x_slice.set_time_step(0)
            avg_vel_x_slice.set_name("average_velocity_x_slice")
            avg_vel_x_slice.save_to_file("avg_vel_x_slice")
            avg_vel_x_slice.plot()

        vel.set_time_step(i)
        vel.set_name("velocity")

        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[2].plot_3d(2)

    vel_base_lam = VectorField(
        [
            PhysicalField.FromFunc(domain, lambda X: 1.0 * (1 - X[1] ** 2) + 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
        ]
    )
    noise_field = (
        FourierField.FromWhiteNoise(coarse_domain, 1e-4)
        .project_onto_domain(domain)
        .no_hat()
    )
    U = vel_base_lam + VectorField([noise_field for _ in range(3)])
    # U = vel_base_lam
    nse = NavierStokesVelVort.FromVelocityField(U, Re=Re, dt=dt)
    nse.end_time = end_time

    # nse.deactivate_jit()
    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()

    nse.set_post_process_fn(post_process)
    nse.post_process()


def run_ld_2021(
    turb: float = 1.0,
    Re_tau: float = 180,
    Nx: int = 28,
    Ny: int = 129,
    Nz: int = 24,
    number_of_steps: int = 10,
    min_number_of_optax_steps: int = -1,
    e_0: float = 1e-3,
    init_file: Optional[str] = None,
) -> None:
    Re_tau = float(Re_tau)
    turb = float(turb)
    assert turb >= 0.0 and turb <= 1.0, "turbulence parameter must be between 0 and 1."
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    aliasing = 3 / 2
    # aliasing = 2
    # aliasing = 1
    e_0 = float(e_0)

    Equation.initialize()

    # max_cfl = 0.65
    max_cfl = 0.10
    end_time = 0.35  # the target time (in ld2021 units)

    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=aliasing,
        dealias_nonperiodic=False,
    )

    coarse_domain = PhysicalDomain.create(
        (Nx, Ny - Ny // 3, Nz),
        # (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=1,
    )
    # coarse_domain = domain
    avg_vel_coeffs = np.loadtxt(
        "./profiles/Re_tau_180_90_small_channel.csv", dtype=np.float64
    )

    def get_vel_field(
        domain: PhysicalDomain, cheb_coeffs: "np_jnp_array"
    ) -> Tuple[VectorField[PhysicalField], "np_jnp_array", "jsd_float"]:
        Ny = domain.number_of_cells(1)
        U_mat = np.zeros((Ny, len(cheb_coeffs)))
        for i in range(Ny):
            for j in range(len(cheb_coeffs)):
                U_mat[i, j] = cheb(j, 0)(domain.grid[1][i])
        U_y_slice = U_mat @ cheb_coeffs
        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = np.moveaxis(
            np.tile(np.tile(U_y_slice, reps=(nz, 1)), reps=(nx, 1, 1)), 1, 2
        )
        max = np.max(u_data)
        vel_base = VectorField(
            [
                PhysicalField(domain, jnp.asarray(u_data)),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            ]
        )
        return vel_base, U_y_slice, max

    vel_base_turb, _, max = get_vel_field(domain, avg_vel_coeffs)
    vel_base_turb = vel_base_turb.normalize_by_max_value()
    u_max_over_u_tau = max
    h_over_delta: float = (
        1.0  # confusingly, LD2021 use channel half-height but call it channel height
    )
    vel_base_lam = VectorField(
        [
            PhysicalField.FromFunc(domain, lambda X: 1.0 * (1 - X[1] ** 2) + 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]),
        ]
    )

    vel_base = (
        turb * vel_base_turb + (1 - turb) * vel_base_lam
    )  # continuously blend from turbulent to laminar mean profile
    vel_base.set_name("velocity_base")

    Re = Re_tau * u_max_over_u_tau / h_over_delta
    # end_time_ = round(end_time * h_over_delta * u_max_over_u_tau)
    end_time_ = cast(float, end_time * h_over_delta * u_max_over_u_tau)

    dt = Equation.find_suitable_dt(domain, max_cfl, (1.0, 1e-5, 1e-5), end_time_)

    print_verb(
        "end time in LD2021 units:", end_time_ / (h_over_delta * u_max_over_u_tau)
    )
    print_verb("end time in dimensional units:", end_time_)
    print_verb("Re:", Re)

    if init_file is None:
        number_of_modes = 60
        n = 64
        lsc_domain = PhysicalDomain.create(
            (2, n, 2),
            (True, False, True),
            scale_factors=domain.scale_factors,
            aliasing=1,
        )
        _, U_base, _ = get_vel_field(lsc_domain, avg_vel_coeffs)
        U_base = U_base / np.max(U_base)
        lsc = LinearStabilityCalculation(
            Re=Re,
            alpha=2 * jnp.pi / 1.87,
            beta=0,
            n=n,
            U_base=cast("np_float_array", U_base),
        )

        v0_0 = lsc.calculate_transient_growth_initial_condition(
            # coarse_domain,
            domain,
            end_time_,
            number_of_modes,
            recompute_full=True,
            save_final=False,
        )
        print_verb(
            "expected gain:",
            lsc.calculate_transient_growth_max_energy(end_time_, number_of_modes),
        )
        v0_0.normalize_by_energy()
        v0_0 *= e_0
        vel_hat = v0_0.hat()
        vel_hat.set_name("velocity_hat")
    else:
        vel_hat = None

    run_input_initial = init_file or vel_hat
    assert run_input_initial is not None

    # assert vel_hat is not None
    # print("vel_coarse")
    # vel_hat.no_hat().plot_3d(2)
    # vel_hat_coarse = vel_hat.project_onto_domain(coarse_domain)
    # vel_hat_coarse.set_name("vel_hat_coarse")
    # vel_hat_coarse.no_hat().plot_3d(2)
    # print("vel_fine")
    # vel_hat_fine = vel_hat_coarse.project_onto_domain(domain)
    # vel_hat_fine.set_name("vel_hat_fine")
    # vel_hat_fine.no_hat().plot_3d(2)
    # print("vel_filtered")
    # print(vel_hat[0].data.shape)
    # vel_hat_filtered = VectorField(
    #     [
    #         FourierField(domain, domain.hat().filter_field(vel_hat[i].data))
    #         for i in range(3)
    #     ]
    # )
    # vel_hat_filtered.set_name("vel_hat_filtered")
    # vel_hat_filtered.no_hat().plot_3d(2)
    # print("vel_hat.no_hat().energy()")
    # print(vel_hat.no_hat().energy())
    # raise Exception("break")

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        # time = (i / (n_steps - 1)) * end_time
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vel_base_hat = nse.get_initial_field("velocity_base_hat")
        vel_base = vel_base_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")

        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[2].plot_3d(2)
        vel[0].plot_3d(0)
        vel[1].plot_3d(0)
        vel[2].plot_3d(0)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)
        vel[0].plot_isosurfaces(0.4)

        vel_total = vel + vel_base
        vel_total.set_name("velocity_total")
        vel_total[0].plot_3d(0)
        vel_total[0].plot_isosurfaces(0.4)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time_
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")

    def run_case(U_hat: VectorField[FourierField], out: bool = False) -> "jsd_float":

        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0
        # U_norm.set_name("vel_norm")
        # U_norm.plot_3d(2)
        nse = NavierStokesVelVortPerturbation.FromVelocityField(
            U_norm, Re=Re, dt=dt, velocity_base_hat=vel_base.hat()
        )
        energy_0_ = U_norm.energy()
        nse.activate_jit()
        nse.end_time = end_time_
        if out:
            nse.write_intermediate_output = True
            nse.set_post_process_fn(post_process)
        else:
            nse.write_intermediate_output = False
        nse.solve()
        if out:
            nse.post_process()
        vel_final = nse.get_latest_field("velocity_hat").no_hat()
        gain = vel_final.energy() / energy_0_
        return gain

    optimiser = OptimiserFourier(
        domain,
        coarse_domain,
        run_case,
        run_input_initial,
        minimise=False,
        force_2d=False,
        # max_iter=number_of_steps,
        max_iter=-1,  # suppress initialisation of optimiser
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
        add_noise=True,
        # add_noise=False,
        noise_amplitude=1e-6,
        learning_rate=1e-6,
    )
    # optimiser.optimise()
    gain, corr = jax.value_and_grad(optimiser.run_fn)(optimiser.parameters)
    print_verb(corr)
    print_verb(gain)


def run_white_noise() -> None:

    Equation.initialize()
    Re = 3000
    e_0 = 1e-4
    Nx, Ny, Nz = 44, 129, 36
    max_cfl = 0.4
    end_time = 5e-1

    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=3 / 2,
        # aliasing=1,
    )
    coarse_domain = PhysicalDomain.create(
        (28, Ny - 20, 24),
        # (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=1,
    )
    dt = Equation.find_suitable_dt(domain, max_cfl, (1.0, 1e-5, 1e-5), end_time)

    velocity_x_base = PhysicalField.FromFunc(
        domain,
        lambda X: 1.0 * (1 - X[1] ** 2) + 0.0 * X[0] * X[2],
        name="velocity_x_base",
    )
    velocity_y_base = PhysicalField.FromFunc(
        domain,
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_y_base",
    )
    velocity_z_base = PhysicalField.FromFunc(
        domain,
        lambda X: 0.0 * X[0] * X[1] * X[2],
        name="velocity_z_base",
    )
    velocity_base_hat = VectorField(
        [velocity_x_base.hat(), velocity_y_base.hat(), velocity_z_base.hat()]
    )
    velocity_base_hat.set_name("velocity_base_hat")

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")

        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vel[2].plot_3d(2)
        vel[0].plot_3d(0)
        vel[1].plot_3d(0)
        vel[2].plot_3d(0)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []

        fig_ = figure.Figure()
        ax_ = fig_.subplots(1, 2)
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

            assert type(ax_) is np.ndarray
            ax_[0].plot(jnp.abs(vel_hat_[2].data[:, Ny // 2, -1]), "o")
            ax_[1].plot(jnp.abs(vel_hat_[2].data[-1, Ny // 2, :]), "o")
            fig_.savefig("plots/spectrum_t" + ".png")

        energy_t_arr = np.array(energy_t)
        print_verb("energy_t", energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")

        fig_ = figure.Figure()
        ax_ = fig_.subplots(1, 2)
        # assert type(ax) is Axes
        assert type(ax_) is np.ndarray
        ax_[0].plot(jnp.abs(vel_hat[2].data[:, Ny // 2, -1]))
        ax_[1].plot(jnp.abs(vel_hat[2].data[-1, Ny // 2, :]))
        print_verb(
            "conti_error (iteration",
            i,
            ")",
            vel_hat.div().no_hat().energy() / vel.energy(),
        )
        fig_.savefig("plots/spectrum_t_" + "{:06}".format(i) + ".png")

    vel_hat: VectorField[FourierField] = VectorField(
        [
            FourierField.FromWhiteNoise(coarse_domain, energy_norm=e_0)
            for _ in coarse_domain.all_dimensions()
        ],
        name="velocity_hat",
    )

    optimiser = OptimiserFourier(
        domain,
        # domain,
        coarse_domain,
        # coarse_domain,
        lambda vel_hat_, t: 1.0,
        vel_hat,
        minimise=False,
        force_2d=False,
        objective_fn_name="gain",
        add_noise=False,
        noise_amplitude=1e-3,
    )
    fig = figure.Figure()
    ax = fig.subplots(1, 2)
    # assert type(ax) is Axes
    assert type(ax) is np.ndarray
    # ax[0].plot(jnp.abs(vel_hat[0].data[:, Ny // 2, 0]), "o")
    # ax[1].plot(jnp.abs(vel_hat[0].data[0, Ny // 2, :]), "o")
    vel_hat.no_hat().plot_3d(2)
    print_verb(
        "conti_error (before)",
        vel_hat.div().no_hat().energy() / vel_hat.no_hat().energy(),
    )
    vel_hat = optimiser.parameters_to_run_input(
        optimiser.run_input_to_parameters(vel_hat)
    )  # should lower conti error
    vel_hat.set_name("velocity_hat")
    print_verb(
        "conti_error (after)",
        vel_hat.div().no_hat().energy() / vel_hat.no_hat().energy(),
    )
    vel_hat.div().no_hat().plot_3d(2)
    vel_hat.div().no_hat().plot_3d(1)
    vel_hat.div().no_hat().plot_3d(0)
    vel_hat[0].diff(0).no_hat().plot_3d(0)
    vel_hat[1].diff(1).no_hat().plot_3d(0)
    vel_hat[2].diff(2).no_hat().plot_3d(0)
    (vel_hat[0].diff(0) + vel_hat[2].diff(2)).no_hat().plot_3d(0)
    # raise Exception("break")
    # ax[0].plot(jnp.abs(vel_hat[0].data[:, Ny // 2, 0]), "o")
    # ax[1].plot(jnp.abs(vel_hat[0].data[0, Ny // 2, :]), "o")
    # vel_hat.no_hat().plot_3d(2)
    # fig.savefig("plots/spectrum.png")
    U = vel_hat.project_onto_domain(domain).no_hat()
    U.update_boundary_conditions()
    # U[1].data = jnp.zeros_like(U[1].data)
    U_norm = U.normalize_by_energy()
    # U_norm *= jnp.sqrt(e_0)
    U_norm *= e_0
    nse = NavierStokesVelVortPerturbation.FromVelocityField(
        U_norm, Re=Re, dt=dt, velocity_base_hat=velocity_base_hat
    )
    # energy_0_ = U_norm.energy()
    nse.set_linearize(False)
    nse.activate_jit()
    # nse.end_time = dt
    nse.end_time = end_time
    vel_hat = nse.get_latest_field("velocity_hat")
    energy_0 = vel_hat.no_hat().energy()
    # ax[0].plot(jnp.abs(vel_hat[0].data[:, Ny // 2, 0]), "o")
    # ax[1].plot(jnp.abs(vel_hat[0].data[0, Ny // 2, :]), "o")
    # nse.perform_time_step()
    # nse.perform_time_step()
    # nse.perform_time_step()

    # vel_hat = nse.get_latest_field("velocity_hat")
    # ax[0].plot(jnp.abs(vel_hat[0].data[:, Ny // 2, 0]), "o")
    # ax[1].plot(jnp.abs(vel_hat[0].data[0, Ny // 2, :]), "o")
    # fig.savefig("plots/spectrum.png")
    nse.write_intermediate_output = True
    nse.set_post_process_fn(post_process)
    nse.solve()
    nse.post_process()
    vel_final = nse.get_latest_field("velocity_hat").no_hat()
    gain = vel_final.energy() / energy_0
    print_verb("gain:", gain)

    # nse_small_dt = NavierStokesVelVortPerturbation.FromVelocityField(
    #     U_norm, Re=Re, dt=dt/10, velocity_base_hat=velocity_base_hat
    # )
    # energy_0_ = U_norm.energy()
    # nse_small_dt.set_linearize(False)
    # nse_small_dt.activate_jit()
    # nse_small_dt.end_time = dt
    # nse_small_dt.write_intermediate_output = True
    # nse_small_dt.set_post_process_fn(post_process)
    # nse_small_dt.solve()

    # vel_final_ = nse_small_dt.get_latest_field("velocity_hat").no_hat()
    # gain_ = vel_final_.energy() / energy_0_
    # print_verb("gain:", gain_)

    # vel_diff = vel_final - vel_final_
    # vel_diff.set_name("vel_diff")
    # vel_diff.plot_3d(0)
    # vel_diff.plot_3d(2)


def run_optimisation_transient_growth_dual(
    Re: float = 3000.0,
    T: float = 15,
    Nx: int = 8,
    Ny: int = 90,
    Nz: int = 8,
    number_of_steps: int = 20,
    min_number_of_optax_steps: int = -1,
) -> None:
    Re = float(Re)
    T = float(T)
    alpha = 1.0
    beta = 0.0

    Equation.initialize()
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    dt = 1e-2
    end_time = T
    number_of_modes = 20  # deliberately low value so that there is room for improvement
    # number_of_modes = 60
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)
    # aliasing = 3 / 2
    aliasing = 1

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    domain: PhysicalDomain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=scale_factors,
        aliasing=aliasing,
    )

    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        recompute_full=True,
        save_final=False,
    )

    e_0 = 1e-6
    v0_0_norm = v0_0.normalize_by_energy()
    v0_0_norm *= e_0
    v0_0_hat = v0_0_norm.hat()

    def post_process(nse: NavierStokesVelVortPerturbation, i: int) -> None:
        n_steps = nse.get_number_of_fields("velocity_hat")
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()

        vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vort.set_time_step(i)
        vort.set_name("vorticity")
        vel[0].plot_3d(2)
        vel[1].plot_3d(2)
        vort[2].plot_3d(2)
        vel.plot_streamlines(2)
        vel[0].plot_isolines(2)

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ts = []
        energy_t = []
        for j in range(n_steps):
            time_ = (j / (n_steps - 1)) * end_time
            vel_hat_ = nse.get_field("velocity_hat", j)
            vel_ = vel_hat_.no_hat()
            vel_energy_ = vel_.energy()
            ts.append(time_)
            energy_t.append(vel_energy_)

        energy_t_arr = np.array(energy_t)
        ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
        ax.plot(
            ts[: i + 1],
            energy_t_arr[: i + 1] / energy_t_arr[0],
            "bo",
            label="energy gain",
        )
        fig.legend()
        fig.savefig("plots/plot_energy_t_" + "{:06}".format(i) + ".png")
        fig.savefig("plots/plot_energy_t_final.png")

    v0_hat = v0_0_hat
    old_gain = None

    for i in range(number_of_steps):
        print_verb("iteration", i + 1, "of", number_of_steps)
        v0_hat.set_name("velocity_hat")
        nse = NavierStokesVelVortPerturbation(v0_hat, Re=Re, dt=dt)
        nse.end_time = end_time
        gain, corr = perform_step_navier_stokes_perturbation_dual(nse)
        v0_hat = v0_hat - 1e-3 * corr

        print_verb("")
        print_verb("gain:", gain)
        if old_gain is not None:
            print_verb("gain change:", gain - old_gain)
        print_verb("")

        v0 = v0_hat.no_hat()
        v0.set_time_step(i)
        v0.plot_3d(0)
        v0.plot_3d(2)
        old_gain = gain

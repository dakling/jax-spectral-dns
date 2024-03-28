#!/usr/bin/env python3

import sys
import jax
import scipy.optimize as sciopt
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import matplotlib.figure as figure
from functools import partial
import typing
import time

from jax_spectral_dns.cheb import cheb
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import FourierField, PhysicalField, FourierFieldSlice, VectorField
from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.navier_stokes import NavierStokesVelVort, solve_navier_stokes_laminar
from jax_spectral_dns.navier_stokes_perturbation import (
    NavierStokesVelVortPerturbation,
    solve_navier_stokes_perturbation,
)
from jax_spectral_dns.linear_stability_calculation import LinearStabilityCalculation
from jax_spectral_dns.optimiser import Optimiser, OptimiserPertAndBase

NoneType = type(None)

def run_navier_stokes_turbulent_pseudo_2d():
    Re = 5000

    end_time = 5

    use_antialias = False
    # use_antialias = True # also works fine
    if use_antialias:
        aliasing = 3/2
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
        aliasing=aliasing
    )
    u_fn = lambda X: 1-X[1]**2 + 0.1*(jnp.sin(X[0])*(jnp.cos(jnp.pi/2 * X[1])*(-2*X[1]) - jnp.pi/2 * jnp.sin(jnp.pi/2 * X[1])*(1-X[1]**2))) + 0 * X[2]
    v_fn = lambda X: 0.1*(-jnp.cos(X[0])*(1-X[1]**2)*jnp.cos(jnp.pi/2*X[1])) + 0 * X[2]

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
    nse.before_time_step_fn = None
    nse.after_time_step_fn = None
    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()

    n_steps = len(nse.get_field("velocity_hat"))

    def post_process(nse, i):
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

    nse.post_process_fn = post_process
    nse.post_process()

    energy_t = np.array(energy_t)
    fig = figure.Figure()
    ax = fig.subplots(2, 1)
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
    ax[0].plot(ts, energy_t, "--", label="jax")
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
    ax[1].plot(ts, energy_t / energy_t[0], "--")
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$E$")
    ax[1].set_ylabel("$E/E_0$")
    fig.legend()
    fig.savefig("plots/energy.png")

def run_navier_stokes_turbulent():
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
    nse.before_time_step_fn = None
    nse.after_time_step_fn = None
    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()

    n_steps = len(nse.get_field("velocity_hat"))

    def post_process(nse, i):
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

    nse.post_process_fn = post_process
    nse.post_process()

    energy_t = np.array(energy_t)
    fig = figure.Figure()
    ax = fig.subplots(2, 1)
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
    ax[0].plot(ts, energy_t, "o", label="jax")
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
    ax[1].plot(ts, energy_t / energy_t[0], "o")
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$E$")
    ax[1].set_ylabel("$E/E_0$")
    fig.legend()
    fig.savefig("plots/energy.png")


def run_pseudo_2d():
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
    vel_x_hat = nse.get_initial_field("velocity_hat")

    eps = 5e-3
    nse.init_velocity(vel_x_hat + (u * eps).hat())

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.get_physical_domain())
    energy_over_time_fn = lambda t: eps**2 * energy_over_time_fn_raw(t)
    energy_x_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.get_physical_domain()
    )[0](t, 0)
    energy_y_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.get_physical_domain()
    )[0](t, 1)
    print_verb("eigenvalue: ", ev)
    plot_interval = 10

    vel_pert_0 = nse.get_initial_field("velocity_hat").no_hat()[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    energy_x_t = []
    energy_y_t = []
    energy_t_ana = []
    energy_x_t_ana = []
    energy_y_t_ana = []

    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            # vel_hat = nse.get_field("velocity_hat", i)
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            vel_x_max = vel[0].max()
            print_verb("vel_x_max: ", vel_x_max)
            vel_x_fn_ana = (
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
            vel_pert_energy = 0
            v_1_lap_p = nse.get_latest_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p_0 = nse.get_initial_field("v_1_lap_hat_p").no_hat()
            v_1_lap_p.time_step = i
            v_1_lap_p.plot_3d(2)
            v_1_lap_p.plot_center(0)
            v_1_lap_p.plot_center(1, v_1_lap_p_0)
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
            ts.append(nse.time)
            energy_t.append(vel_pert_energy)
            energy_x_t.append(vel_pert[0].energy())
            energy_y_t.append(vel_pert[1].energy())
            energy_t_ana.append(energy_over_time_fn(nse.time))
            energy_x_t_ana.append(energy_x_over_time_fn(nse.time))
            energy_y_t_ana.append(energy_y_over_time_fn(nse.time))
            # if i > plot_interval * 3:
            if True:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_t_ana)
                ax.plot(ts, energy_t, ".")
                fig.savefig("plots/energy_t.png")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_x_t_ana)
                ax.plot(ts, energy_x_t, ".")
                fig.savefig("plots/energy_x_t.png")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_y_t_ana)
                ax.plot(ts, energy_y_t, ".")
                fig.savefig("plots/energy_y_t.png")
        # input("carry on?")

    nse.after_time_step_fn = None
    nse.before_time_step_fn = before_time_step
    # nse.before_time_step_fn = None

    nse.solve()


def run_dummy_velocity_field():
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

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = PhysicalField.FromFunc(
        nse.get_physical_domain(), vel_x_fn_ana, name="vel_x_ana"
    )

    def after_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            # vel = nse.get_field("velocity_hat", i).no_hat()
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
            vel_pert_energy = 0
            vel_pert_abs = 0
            for j in range(3):
                vel_pert_energy += vel_pert[j].energy()
                vel_pert_abs += abs(vel_pert[j])
            print_verb("velocity perturbation energy: ", vel_pert_energy)
            print_verb("velocity perturbation abs: ", vel_pert_abs)

    nse.after_time_step_fn = after_time_step
    # nse.after_time_step_fn = None
    nse.solve()
    return nse.get_latest_field("velocity_hat").no_hat().field


def run_pseudo_2d_perturbation(
    Re=3000.0,
    alpha=1.02056,
    end_time=1.0,
    dt=1e-2,
    Nx=4,
    Ny=96,
    Nz=2,
    eps=1e-0,
    linearize=True,
    plot=True,
    save=True,
    v0=None,
    aliasing=1.0,
    rotated=False,
    jit=True,
):
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
            rotated=True,
        )

    nse.set_linearize(linearize)
    # nse.initialize()

    if type(v0) == NoneType:
        U = lsc.velocity_field_single_mode(nse.get_physical_domain(), save=save)
    else:
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
    ts = []
    energy_t = []
    energy_x_t = []
    energy_y_t = []
    energy_t_ana = []
    energy_x_t_ana = []
    energy_y_t_ana = []

    def save_array(arr, filename):
        if save:
            f = Path(filename)
            try:
                f.unlink()
            except FileNotFoundError:
                pass
            np.array(arr).dump(filename)

    nse.before_time_step_fn = None
    nse.after_time_step_fn = None

    if jit:
        nse.activate_jit()
    else:
        nse.deactivate_jit()
    nse.write_intermediate_output = plot
    nse.solve()
    nse.deactivate_jit()

    n_steps = len(nse.get_field("velocity_hat"))
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
        ts.append(time)
        energy_t.append(vel_pert_energy)
        energy_x_t.append(vel[0].energy())
        energy_y_t.append(vel[1].energy())
        energy_t_ana.append(energy_over_time_fn(time))
        energy_x_t_ana.append(energy_over_time_fn(time, 0))
        energy_y_t_ana.append(energy_over_time_fn(time, 1))

    vel_pert = nse.get_latest_field("velocity_hat").no_hat()
    # vel_pert_old = nse.get_field("velocity_hat", nse.time_step - 3).no_hat()
    vel_pert_energy = vel_pert.energy()
    # vel_pert_energy_old = vel_pert_old.energy()

    fig = figure.Figure()
    ax = fig.subplots(1, 1)
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


def run_jimenez_1990(start_time=0):
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

    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            vel_pert = nse.get_latest_field("velocity_hat").no_hat()
            vel_base = nse.get_latest_field("velocity_base_hat").no_hat()
            vel = vel_base + vel_pert
            vort = vel.curl()
            vort.set_name("vorticity")
            vort.set_time_step(i)
            vel_moving_frame = vel.shift([-0.353, 0, 0])
            vel_moving_frame.set_name("velocity_moving_frame")
            vel_moving_frame.set_time_step(i)
            vel_moving_frame.plot_streamlines(2)
            # remove old fields
            [
                f.unlink()
                for f in Path("./fields/").glob(
                    "velocity_perturbation_0_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            [
                f.unlink()
                for f in Path("./fields/").glob(
                    "velocity_perturbation_1_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            [
                f.unlink()
                for f in Path("./fields/").glob(
                    "velocity_perturbation_2_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            for j in range(3):
                vel_pert[j].save_to_file(
                    "velocity_perturbation_" + str(j) + "_t_" + str(i)
                )
                vort[j].plot_3d()
                vort[j].plot_3d(2)
                vort[j].plot_isolines(2)
                # vel_moving_frame[j].plot_3d(2)

    nse.before_time_step_fn = before_time_step
    nse.solve()


def run_transient_growth_nonpert(
    Re=3000.0,
    T=15.0,
    alpha=1.0,
    beta=0.0,
    end_time=None,
    eps=1e-3,
    Nx=4,
    Ny=50,
    Nz=4,
    plot=True,
):

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
        aliasing=3 / 2,
    )
    # nse.initialize()

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

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

    e_max = lsc.calculate_transient_growth_max_energy(
        nse.get_physical_domain(), T, number_of_modes
    )

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    if plot and abs(Re - 3000) < 1e-3:
        rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    # nse.before_time_step_fn = lambda nse: post_process(nse, nse.time_step)
    nse.before_time_step_fn = None
    nse.after_time_step_fn = None
    nse.post_process_fn = post_process

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
    Re=3000.0,
    T=15.0,
    alpha=1.0,
    beta=0.0,
    end_time=None,
    eps=1e-5,
    Nx=4,
    Ny=50,
    Nz=4,
    linearize=True,
    plot=True,
):

    # ensure that these variables are not strings as they might be passed as command line arguments
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)
    if type(linearize) == str:
        linearize = linearize == "True"

    eps = float(eps)

    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)

    if end_time is None:
        end_time = T
    else:
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
    )
    # nse.initialize()

    nse.set_linearize(linearize)

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

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

    e_max = lsc.calculate_transient_growth_max_energy(
        nse.get_physical_domain(), T, number_of_modes
    )

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    if plot and abs(Re - 3000) < 1e-3:
        rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    # nse.before_time_step_fn = lambda nse: post_process(nse, nse.time_step)
    nse.before_time_step_fn = None
    nse.after_time_step_fn = None
    nse.post_process_fn = post_process

    nse.activate_jit()
    nse.write_intermediate_output = plot
    nse.solve()
    nse.deactivate_jit()
    nse.post_process()

    energy_t = np.array(energy_t)
    if plot:
        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        ax.plot(ts, energy_t / energy_t[0], ".", label="growth (DNS)")
        if abs(Re - 3000) < 1e-3:
            ax.plot(
                rh_93_data[0],
                rh_93_data[1],
                "--",
                label="growth (Reddy/Henningson 1993)",
            )
        fig.legend()
        fig.savefig("plots/energy_t.png")

    gain = energy_t[-1] / energy_t[0]
    print_verb("final energy gain:", gain)
    print_verb("expected final energy gain:", e_max)

    return (gain, e_max, ts, energy_t)


def run_transient_growth_time_study(transient_growth_fn=run_transient_growth):

    if type(transient_growth_fn) is str:
        transient_growth_fn = globals()[sys.argv[2]]
    Re = 3000

    rh_93_data = np.genfromtxt("rh93_transient_growth.csv", delimiter=",").T

    fig = figure.Figure()
    ax = fig.subplots(1, 1)
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
    Re=3000.0,
    T=15,
    Nx=8,
    Ny=90,
    Nz=8,
    number_of_steps=20,
    min_number_of_optax_steps=-1,
):
    Re = float(Re)
    T = float(T)
    alpha=1.0
    beta=0.0

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
    v0_0_hat = v0_0_norm.hat()

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    def run_case(U_hat, out=False):

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
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = Optimiser(
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


def run_optimisation_transient_growth_y_profile(
    Re=3000.0,
    T=15,
    Nx=8,
    Ny=90,
    Nz=8,
    number_of_steps=20,
    min_number_of_optax_steps=4,
):
    Re = float(Re)
    T = float(T)
    alpha=1.0
    beta=0.0
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

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    def run_case(U_hat, out=False):
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
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    def run_input_to_params(vel_hat):
        v0_1 = vel_hat[1].data[1, :, 0] * (1 + 0j)
        v0_0_00_hat = vel_hat[0].data[0, :, 0] * (1 + 0j)
        v0 = tuple([v0_1, v0_0_00_hat])
        return v0

    def params_to_run_input(params):
        v1_yslice = params[0]
        v1_hat = domain.field_hat(lsc.y_slice_to_3d_field(domain, v1_yslice))
        v0_00 = params[1]
        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
            domain, None, v1_hat, v0_00, None, two_d=True
        )
        U_hat = VectorField.FromData(FourierField, domain, U_hat_data)
        return U_hat

    optimiser = Optimiser(
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
    )
    optimiser.optimise()


def run_optimisation_transient_growth_nonlinear(
    Re=3000.0,
    T=15,
    Nx=48,
    Ny=64,
    Nz=8,
    number_of_steps=20,
    min_number_of_optax_steps=-1,
        e_0=1e-3
):
    Re = float(Re)
    T = float(T)
    alpha=1.0
    beta=0.0

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

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    def run_case(U_hat, out=False):

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
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = Optimiser(
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
    Re=3000.0,
    T=15,
    Nx=48,
    Ny=64,
    Nz=48,
    number_of_steps=20,
    min_number_of_optax_steps=-1,
        e_0=1e-3,
        init_file=None
):
    Re = float(Re)
    T = float(T)
    alpha=1.0
    beta=0.0

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
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 1.0)
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


    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
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

    def run_case(U_hat, out=False):

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
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    optimiser = Optimiser(
        domain,
        run_case,
        init_file or v0_0_hat,
        minimise=False,
        force_2d=False,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
    )
    optimiser.optimise()

def run_optimisation_transient_growth_mean_y_profile(
    Re=3000.0,
    T=15,
    Nx=8,
    Ny=90,
    Nz=8,
    number_of_steps=20,
    min_number_of_optax_steps=-1,
):
    Re = float(Re)
    T = float(T)
    alpha=1.0
    beta=0.0
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
    velocity_base = VectorField(
        [velocity_x_base, velocity_y_base, velocity_z_base]
    )
    velocity_base_hat = velocity_base.hat()
    velocity_base_hat.set_name("velocity_base_hat")

    def post_process(nse, i):
        if i == 0:
            vel_base = nse.get_initial_field("velocity_base_hat").no_hat()
            vel_base.plot_3d(2)
        n_steps = len(nse.get_field("velocity_hat"))
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

    def run_case(inp, out=False):
        U_hat, U_base_hat = inp
        U_base = U_base_hat.no_hat()
        U_base.normalize_by_max_value()
        U_base_hat = U_base.hat()
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, dt=dt, Re=Re, velocity_base_hat=U_base_hat)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        if out:
            nse.write_intermediate_output = True
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None
        if out:
            nse.post_process()

        gain = vel.energy() / vel_0.energy()
        return gain

    def run_input_to_params(inp):
        vel_hat, vel_base = inp
        v0_1 = vel_hat[1].data[1, :, 0] * (1 + 0j)
        v0_0_00_hat = vel_hat[0].data[0, :, 0] * (1 + 0j)

        # optimise entire slice
        # v0_base_hat = vel_base[0].get_data()[0, :, 0]

        # optimise using phi_s basis
        # v0_base_hat_coeffs = jnp.array([-0.5+0j, 0.0+0j, 0.0+0j])

        # optimise using parametric profile
        v0_base_hat_coeffs = jnp.array([jnp.log(2.0+0j), jnp.log(1.0+0j)])

        v0 = tuple([v0_1, v0_0_00_hat, v0_base_hat_coeffs])
        return v0

    def params_to_run_input(params):
        v1_yslice = params[0]
        v1_hat = domain.field_hat(lsc.y_slice_to_3d_field(domain, v1_yslice))
        v0_00 = params[1]
        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
            domain, None, v1_hat, v0_00, None, two_d=True
        )

        # optimise entire slice
        # v0_base_yslice = params[2]
        # v0_base_hat = domain.field_hat(lsc0.y_slice_to_3d_field(domain, v0_base_yslice))

        # optimise using phi_s basis
        v0_base_yslice_coeffs = params[2]
        v0_base_zeros = jnp.zeros_like(v0_base_yslice_coeffs)
        for _ in range(3):
            v0_base_yslice_coeffs = jnp.concatenate((v0_base_yslice_coeffs, v0_base_zeros))
        v0_base_hat = (lsc0.velocity_field(domain, v0_base_yslice_coeffs, symm=True)).hat()[0].get_data()

        # optimise using parametric profile
        v0_base_yslice = params[2]
        m = jnp.exp(v0_base_yslice_coeffs[0].real) # ensures > 0
        n = jnp.exp(v0_base_yslice_coeffs[1].real) # ensures > 0
        print_verb("m, n:", m, n)
        v0_base_yslice = jnp.array(list(map(lambda y: (1 - y**(m))**(1/n), domain.grid[1])))
        v0_base_hat =  domain.field_hat(lsc0.y_slice_to_3d_field(domain, v0_base_yslice))

        U_hat = VectorField.FromData(FourierField, domain, U_hat_data)
        U_base = VectorField.FromData(FourierField, domain, [v0_base_hat, jnp.zeros_like(v0_base_hat), jnp.zeros_like(v0_base_hat)])
        return (U_hat, U_base)

    optimiser = OptimiserPertAndBase(
        domain,
        run_case,
        (v0_0_hat, velocity_base_hat),
        minimise=False,
        force_2d=True,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        run_input_to_parameters_fn=run_input_to_params,
        parameters_to_run_input_fn=params_to_run_input,
        objective_fn_name="gain",
    )
    optimiser.optimise()


def run_dedalus(Re=3000.0, T=15.0, alpha=1.0, beta=0.0):
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    eps = 1e-5

    Nx = 64
    Ny = 90
    Nz = 24
    # Nx = 4
    # Ny = 50
    # Nz = 4
    end_time = 1.01 * T

    number_of_modes = 80

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

    nse = solve_navier_stokes_perturbation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi),
    )

    # nse.set_linearize(False)
    nse.set_linearize(True)

    U = lsc.calculate_transient_growth_initial_condition(
        nse.get_physical_domain(), T, number_of_modes, recompute_full=True
    )

    eps_ = eps / jnp.sqrt(U.energy())
    U_hat = U.hat() * eps_
    print_verb("U energy norm: ", jnp.sqrt(U.energy()))
    # print_verb("U energy norm (RH): ", jnp.sqrt(U.energy_norm(1)))

    nse.init_velocity(U_hat)

    U_ = U * eps_
    energy0 = U_.energy()
    print_verb("U energy norm (normalized): ", energy0)
    # U_ = U
    for i in range(3):
        U_[i].name = "uvw"[i]
        # U_hat[i].name = "uvw"[i]
        U_[i].save_to_file("uvw"[i])
        # print_verb(i, U_[i])
        # U_hat[i].save_to_file("uvw"[i])
    # U_.plot_streamlines(2)
    U_[0].plot_3d(2)
    U_[1].plot_3d(2)
    U_[2].plot_3d(2)
    U_.plot_streamlines(2)
    U_.plot_vectors(2)

    plot_interval = 5

    def post_process(nse, i):
        if i % plot_interval == 0:
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            vel_pert = VectorField([vel[0], vel[1], vel[2]])
            # vel_pert_old = nse.get_field("velocity_hat", max(0, i - 1)).no_hat()
            vort = vel.curl()
            for j in range(3):
                vel[j].time_step = i
                vort[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vort[j].name = "vorticity_" + "xyz"[j]
                vel[j].plot_3d()
                vel[j].plot_3d(2)
                vort[j].plot_3d(2)
                vel[j].plot_center(0)
                vel[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            # vel_pert_energy_old = vel_pert_old.energy()
            vel_pert_energy_norm = vel_pert.energy_norm(1)
            print_verb("\n\n")
            print_verb(
                "velocity perturbation energy: ",
                vel_pert_energy,
            )
            print_verb(
                "velocity perturbation relative change: ",
                vel_pert_energy / energy0,
            )
            print_verb(
                "velocity perturbation energy change: ",
                vel_pert_energy - energy0,
            )

    nse.before_time_step_fn = None
    nse.post_process_fn = post_process

    nse.solve()
    nse.post_process()

    U_ = nse.get_latest_field("velocity_hat").no_hat()
    for i in range(3):
        U_[i].name = "uvw"[i]
        # U_hat[i].name = "uvw"[i]
        U_[i].save_to_file("uvw"[i] + "_final")


def run_ld_2020(
    turb=True,
    Re_tau=180,
    Nx=60,
    Ny=90,
    Nz=48,
    number_of_steps=10,
    min_number_of_optax_steps=-1,
):
    Re_tau = float(Re_tau)
    turb = str(turb) == "True"
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)
    number_of_steps = int(number_of_steps)
    min_number_of_optax_steps = int(min_number_of_optax_steps)
    aliasing = 3 / 2
    # aliasing = 1
    Nx = int(Nx * ((3 / 2) / aliasing))
    Nz = int(Nz * ((3 / 2) / aliasing))

    # Equation.initialize()

    dt = 3e-3
    end_time = 0.7  # in ld2020 units
    # end_time = 0.02 # in ld2020 units
    e_0 = 1e-6
    domain = PhysicalDomain.create(
        (Nx, Ny, Nz),
        (True, False, True),
        scale_factors=(1.87, 1.0, 0.93),
        aliasing=aliasing,
    )
    avg_vel_coeffs = np.loadtxt("./profiles/Re_tau_180_90_small_channel.csv")

    def get_vel_field(domain, cheb_coeffs):
        U_mat = np.zeros((Ny, len(cheb_coeffs)))
        for i in range(Ny):
            for j in range(len(cheb_coeffs)):
                U_mat[i, j] = cheb(j, 0, domain.grid[1][i])
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

    if turb:
        print_verb("using turbulent base profile")
        vel_base, _, max = get_vel_field(domain, avg_vel_coeffs)
        vel_base = vel_base.normalize_by_max_value()
        vel_base.set_name("velocity_base")
        u_max_over_u_tau = max
    else:
        print_verb("using laminar base profile")
        vel_base = VectorField(
            [
                PhysicalField.FromFunc(
                    domain, lambda X: 1.0 * (1 - X[1] ** 2) + 0 * X[2]
                ),
                PhysicalField.FromFunc(
                    domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]
                ),
                PhysicalField.FromFunc(
                    domain, lambda X: 0.0 * (1 - X[1] ** 2) + 0 * X[2]
                ),
            ]
        )

        vel_base.set_name("velocity_base")
        u_max_over_u_tau = 18.5  # matches Vilda's profile

    vel_pert = VectorField(
        [
            PhysicalField.FromFunc(
                domain,
                lambda X: 0.1
                * (1 - X[1] ** 2)
                * 0.5
                * (
                    0 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 1.87 * X[0])
                    + 1 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 0.93 * X[2])
                ),
            ),
            PhysicalField.FromFunc(
                domain,
                lambda X: 0.0
                * (1 - X[1] ** 2)
                * 0.5
                * (
                    0.1 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 1.87 * X[0])
                    + 0.1 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 0.93 * X[2])
                ),
            ),
            PhysicalField.FromFunc(
                domain,
                lambda X: 0.01
                * (1 - X[1] ** 2)
                * 0.5
                * (
                    0.1 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 1.87 * X[0])
                    + 0.0 * jnp.cos(1 / 0.5 * 2 * jnp.pi / 0.93 * X[2])
                ),
            ),
        ]
    )
    vel_pert.set_name("velocity")
    vel_pert.normalize_by_energy
    vel_pert *= e_0

    vel_hat = vel_pert.hat()
    vel_hat.set_name("velocity_hat")

    Re = Re_tau * u_max_over_u_tau
    # end_time_ = end_time * u_max_over_u_tau
    end_time_ = round(end_time * u_max_over_u_tau)
    print_verb("end time in LD2020 units:", end_time_ / u_max_over_u_tau)

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
        # time = (i / (n_steps - 1)) * end_time
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

    def run_case(U_hat, out=False):

        U = U_hat.no_hat()
        U.update_boundary_conditions()
        U_norm = U.normalize_by_energy()
        U_norm *= e_0
        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, Re=Re, dt=dt)
        energy_0_ = U_norm.energy()
        nse.activate_jit()
        nse.end_time = end_time_
        if out:
            nse.write_intermediate_output = True
            nse.post_process_fn = post_process
        else:
            nse.write_intermediate_output = False
        nse.solve()
        if out:
            nse.post_process()
        vel_final = nse.get_latest_field("velocity_hat").no_hat()
        gain = vel_final.energy() / energy_0_
        return gain

    optimiser = Optimiser(
        domain,
        run_case,
        vel_hat,
        minimise=False,
        force_2d=False,
        max_iter=number_of_steps,
        use_optax=min_number_of_optax_steps >= 0,
        min_optax_steps=min_number_of_optax_steps,
        objective_fn_name="gain",
    )
    optimiser.optimise()

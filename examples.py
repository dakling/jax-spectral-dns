#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import matplotlib.figure as figure
from numpy import genfromtxt

from importlib import reload
import sys

try:
    reload(sys.modules["domain"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load Domain")
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load Field")
from field import Field, FourierFieldSlice, VectorField

try:
    reload(sys.modules["equation"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load Equation")
from equation import Equation

try:
    reload(sys.modules["navier_stokes"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load Navier Stokes")
from navier_stokes import NavierStokesVelVort, solve_navier_stokes_laminar

try:
    reload(sys.modules["navier_stokes_pertubation"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load navier-stokes-pertubation")
from navier_stokes_pertubation import solve_navier_stokes_pertubation

try:
    reload(sys.modules["linear_stability_calculation"])
except:
    if hasattr(sys, "ps1"):
        print("Unable to load linear stability")
from linear_stability_calculation import LinearStabilityCalculation

NoneType = type(None)

def run_optimization():
    Re = 1e0
    Ny = 24
    end_time = 1

    nse = solve_navier_stokes_laminar(
        Re=Re, Ny=Ny, end_time=end_time, pertubation_factor=0.0
    )

    def run(v0):
        nse_ = solve_navier_stokes_laminar(
            Re=Re, Ny=Ny, end_time=end_time, max_iter=10, pertubation_factor=0.0
        )
        nse_.max_iter = 10
        v0_field = Field(nse_.domain_no_hat, v0)
        vel_0 = nse_.get_initial_field("velocity_hat")
        vel_0_new = VectorField([v0_field.hat(), vel_0[1], vel_0[2]])
        nse_.set_field("velocity_hat", 0, vel_0_new)
        nse_.after_time_step_fn = None
        nse_.solve()
        vel_out = nse_.get_latest_field("velocity_hat").no_hat()
        return (vel_out[0].max() / vel_0[0].max()).real

    vel_x_fn_ana = (
        lambda X: -1 * jnp.pi / 3 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    )
    v0_0 = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana)

    v0s = [v0_0.field]
    eps = 1e3
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_field = Field(nse.domain_no_hat, corr, name="correction")
        corr_field.update_boundary_conditions()
        print("gain: " + str(gain))
        print("corr (abs): " + str(abs(corr_field)))
        v0s.append(v0s[-1] + eps * corr_field.field)
        v0_new = Field(nse.domain_no_hat, v0s[-1])
        v0_new.name = "vel_0_" + str(i)
        v0_new.plot(v0_0)


def run_navier_stokes_turbulent():
    Re = 1.8e6

    end_time = 50
    s_x = 1.87
    s_z = 0.93
    # s_x = 2 * jnp.pi
    # s_z = 2 * jnp.pi
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Ny=60,
        Nx=64,
        end_time=end_time,
        pertubation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, pertubation_factor=1
        # Re=Re, Ny=48, Nx=24, end_time=end_time, pertubation_factor=1
    )

    def vortex_fun(center_y, center_z, a):
        def ret(X):
            _, y, z = X[0], X[1], X[2]
            u = 0
            v = -a * (z - center_z) / s_z
            w = a * (y - center_y) / 2
            return (u, v, w)

        return ret

    def ts_vortex_fun(center_x, center_z, a):
        def ret(X):
            # x,_,z = X
            x, _, z = X[0], X[1], X[2]
            u = a * (z - center_z) / s_z
            v = 0
            w = -a * (x - center_x) / s_x
            return (u, v, w)

        return ret

    vortex_1_fun = vortex_fun(0.0, s_z / 2, 0.1)
    vortex_2_fun = vortex_fun(0.0, -s_z / 2, 0.1)
    ts_vortex_1_fun = ts_vortex_fun(s_x / 2, 0.0, 0.1)
    ts_vortex_2_fun = ts_vortex_fun(-s_x / 2, 0.0, 0.1)
    # vortex_sum = lambda X: vortex_1_fun(X) + vortex_2_fun(X)
    vortex_sum = lambda X: [
        ts_vortex_1_fun(X)[i]
        + ts_vortex_2_fun(X)[i]
        + vortex_1_fun(X)[i]
        + vortex_2_fun(X)[i]
        for i in range(3)
    ]

    # Add small velocity perturbations localized to the shear layers
    omega = 0.05

    Ly = 2.0
    Lz = s_z
    # vel_x_fn = lambda X: jnp.pi / 3 * jnp.cos(
    #     X[1] * jnp.pi / 2) + (1 - X[1]**2) * vortex_sum(X)[0]
    vel_x_fn = (
        lambda X: ((0.5 + X[1] / Ly) * (0.5 - X[1] / Ly)) / 0.25
        + 0.1 * jnp.sin(2 * jnp.pi * X[2] / Lz * omega)
        + 0 * X[0]
    )
    vel_x = Field.FromFunc(nse.domain_no_hat, vel_x_fn, name="velocity_x")

    # vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) * vortex_sum(X)[1]
    vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + 0.1 * jnp.sin(
        2 * jnp.pi * X[2] / Lz * omega
    )
    # vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) *vortex_sum(X)[2]
    vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2]
    vel_y = Field.FromFunc(nse.domain_no_hat, vel_y_fn, name="velocity_y")
    vel_z = Field.FromFunc(nse.domain_no_hat, vel_z_fn, name="velocity_z")
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 2

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel[0].plot_3d()
    vel[1].plot_3d()
    vel[2].plot_3d()

    def after_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_field("velocity_hat", i).no_hat()
            vort_hat, _ = nse.get_vorticity_and_helicity()
            vort = vort_hat.no_hat()
            vel[0].plot_3d()
            vel[1].plot_3d()
            vel[2].plot_3d()
            vort[0].plot_3d()
            vort[1].plot_3d()
            vort[2].plot_3d()
            vel[0].plot_center(1)
            vel[1].plot_center(1)
            vel[2].plot_center(1)

    nse.after_time_step_fn = after_time_step
    # nse.after_time_step_fn = None
    nse.solve()
    return nse.get_latest_field("velocity_hat").no_hat().field


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
    lsc = LinearStabilityCalculation(Re, alpha, Ny)

    end_time = 100
    nse = solve_navier_stokes_laminar(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(4 * (2 * jnp.pi / alpha), 1.0, 1.0),
    )

    u = lsc.velocity_field(nse.domain_no_hat)
    vel_x_hat = nse.get_initial_field("velocity_hat")

    eps = 5e-3
    nse.init_velocity(vel_x_hat + (u * eps).hat())

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.domain_no_hat)
    energy_over_time_fn = lambda t: eps**2 * energy_over_time_fn_raw(t)
    energy_x_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.domain_no_hat
    )[0](t, 0)
    energy_y_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.domain_no_hat
    )[0](t, 1)
    print("eigenvalue: ", ev)
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
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            vel_x_max = vel[0].max()
            print("vel_x_max: ", vel_x_max)
            vel_x_fn_ana = (
                lambda X: -vel_x_max * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            )
            vel_x_ana = Field.FromFunc(
                nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana"
            )
            # vel_1_lap_a = nse.get_field("v_1_lap_hat_a", i).no_hat()
            # vel_1_lap_a.plot_3d()
            vel_pert = VectorField([vel[0] - vel_x_ana, vel[1], vel[2]])
            vel_hat_old = nse.get_field("velocity_hat", max(0, i - 1))
            vel_old = vel_hat_old.no_hat()
            vel_x_max_old = vel_old[0].max()
            vel_x_fn_ana_old = (
                lambda X: -vel_x_max_old * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
            )
            vel_x_ana_old = Field.FromFunc(
                nse.domain_no_hat, vel_x_fn_ana_old, name="vel_x_ana_old"
            )
            vel_pert_old = VectorField(
                [vel_old[0] - vel_x_ana_old, vel_old[1], vel_old[2]]
            )
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
                vel_pert[j].name = "velocity_pertubation_" + "xyz"[j]
                vel_pert[j].plot_3d()
                vel_pert[j].plot_3d(2)
                # vel_pert[j].plot_center(0)
                # vel_pert[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            print(
                "analytical velocity pertubation energy: ",
                energy_over_time_fn(nse.time),
            )
            print("velocity pertubation energy: ", vel_pert_energy)
            print("velocity pertubation energy x: ", vel_pert[0].energy())
            print(
                "analytical velocity pertubation energy x: ",
                energy_x_over_time_fn(nse.time),
            )
            print("velocity pertubation energy y: ", vel_pert[1].energy())
            print(
                "analytical velocity pertubation energy y: ",
                energy_y_over_time_fn(nse.time),
            )
            print("velocity pertubation energy z: ", vel_pert[1].energy())
            vel_pert_energy_old = vel_pert_old.energy()
            print(
                "velocity pertubation energy change: ",
                vel_pert_energy - vel_pert_energy_old,
            )
            print(
                "velocity pertubation energy x change: ",
                vel_pert[0].energy() - vel_pert_old[0].energy(),
            )
            print(
                "velocity pertubation energy y change: ",
                vel_pert[1].energy() - vel_pert_old[1].energy(),
            )
            print(
                "velocity pertubation energy z change: ",
                vel_pert[2].energy() - vel_pert_old[2].energy(),
            )
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
                fig.savefig("plots/energy_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_x_t_ana)
                ax.plot(ts, energy_x_t, ".")
                fig.savefig("plots/energy_x_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_y_t_ana)
                ax.plot(ts, energy_y_t, ".")
                fig.savefig("plots/energy_y_t.pdf")
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
        # pertubation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, pertubation_factor=1
        # Re=Re, Ny=60, Nx=32, end_time=end_time, pertubation_factor=1
        Re=Re,
        Ny=96,
        Nx=64,
        end_time=end_time,
        pertubation_factor=0,
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
    vel_x = Field.FromFunc(nse.domain_no_hat, vel_x_fn, name="velocity_x")
    vel_y = Field.FromFunc(nse.domain_no_hat, vel_y_fn, name="velocity_y")
    vel_z = Field.FromFunc(nse.domain_no_hat, vel_z_fn, name="velocity_z")
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 1

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = Field.FromFunc(nse.domain_no_hat, vel_x_fn_ana, name="vel_x_ana")

    def after_time_step(nse):
        i = nse.time_step
        if (i - 1) % plot_interval == 0:
            vel = nse.get_field("velocity_hat", i).no_hat()
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
            print("velocity pertubation energy: ", vel_pert_energy)
            print("velocity pertubation abs: ", vel_pert_abs)

    nse.after_time_step_fn = after_time_step
    # nse.after_time_step_fn = None
    nse.solve()
    return nse.get_latest_field("velocity_hat").no_hat().field


def run_pseudo_2d_pertubation(
    Re=6000.0,
    alpha=1.02056,
    end_time=10.0,
    Nx=12,
    Ny=50,
    Nz=2,
    eps=1e-0,
    linearize=True,
    plot=True,
    save=True,
    v0=None,
):
    Re = float(Re)
    Nx = int(Nx)
    Ny = int(Ny)
    Nz = int(Nz)

    lsc = LinearStabilityCalculation(Re, alpha, 96)

    nse = solve_navier_stokes_pertubation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-6),
    )

    nse.set_linearize(linearize)

    if type(v0) == NoneType:
        # U = lsc.velocity_field(nse.domain_no_hat).normalize()
        U = lsc.velocity_field(nse.domain_no_hat)
    else:
        # U = VectorField([Field(nse.domain_no_hat, v0[i]) for i in range(3)]).normalize()
        U = VectorField([Field(nse.domain_no_hat, v0[i]) for i in range(3)])
        print(U[0].energy())

    U_hat = U.hat()
    nse.init_velocity(U_hat * eps)


    energy_over_time_fn, _ = lsc.energy_over_time(nse.domain_no_hat, eps=eps)
    plot_interval = 1

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

    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            vel_pert = VectorField([vel[0], vel[1], vel[2]])
            vel_pert_old = nse.get_field("velocity_hat", max(0, i - 1)).no_hat()
            vort = vel.curl()
            for j in range(3):
                vel[j].time_step = i
                vort[j].time_step = i
                vel[j].name = "velocity_" + "xyz"[j]
                vort[j].name = "vorticity_" + "xyz"[j]
                if plot:
                    # vel[j].plot_3d()
                    vel[j].plot_3d(2)
                    vort[j].plot_3d(2)
                    vel[j].plot_center(0)
                    vel[j].plot_center(1)
            vel_pert_energy = vel_pert.energy()
            vel_pert_energy_old = vel_pert_old.energy()
            if plot:
                print("velocity pertubation energy: ", vel_pert_energy)
                print("\n\n")
                print(
                    "velocity pertubation energy change: ",
                    vel_pert_energy - vel_pert_energy_old,
                )
                print(
                    "velocity pertubation energy x change: ",
                    vel_pert[0].energy() - vel_pert_old[0].energy(),
                )
                print(
                    "velocity pertubation energy y change: ",
                    vel_pert[1].energy() - vel_pert_old[1].energy(),
                )
                print(
                    "velocity pertubation energy z change: ",
                    vel_pert[2].energy() - vel_pert_old[2].energy(),
                )
                print("")
            ts.append(nse.time)
            energy_t.append(vel_pert_energy)
            energy_x_t.append(vel_pert[0].energy())
            energy_y_t.append(vel_pert[1].energy())
            energy_t_ana.append(energy_over_time_fn(nse.time))
            energy_x_t_ana.append(energy_over_time_fn(nse.time, 0))
            energy_y_t_ana.append(energy_over_time_fn(nse.time, 1))
            save_array(ts, "fields/ts")
            save_array(energy_t, "fields/energy_Re_" + str(Re))
            save_array(energy_x_t, "fields/energy_x_Re_" + str(Re))
            save_array(energy_y_t, "fields/energy_y_Re_" + str(Re))
            save_array(energy_t_ana, "fields/energy_ana_Re_" + str(Re))
            save_array(energy_x_t_ana, "fields/energy_x_ana_Re_" + str(Re))
            save_array(energy_y_t_ana, "fields/energy_y_ana_Re_" + str(Re))
            if plot:
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_t_ana, label="analytical growth")
                ax.plot(ts, energy_t, ".", label="numerical growth")
                fig.legend()
                fig.savefig("plots/energy_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_x_t_ana, label="analytical growth")
                ax.plot(ts, energy_x_t, ".", label="numerical growth")
                fig.legend()
                fig.savefig("plots/energy_x_t.pdf")
                fig = figure.Figure()
                ax = fig.subplots(1, 1)
                ax.plot(ts, energy_y_t_ana, label="analytical growth")
                ax.plot(ts, energy_y_t, ".", label="numerical growth")
                fig.legend()
                fig.savefig("plots/energy_y_t.pdf")

    nse.before_time_step_fn = before_time_step
    nse.after_time_step_fn = None

    nse.solve()

    vel_pert = nse.get_latest_field("velocity_hat").no_hat()
    vel_pert_old = nse.get_field("velocity_hat", nse.time_step - 3).no_hat()
    vel_pert_energy = vel_pert.energy()
    vel_pert_energy_old = vel_pert_old.energy()
    return (
        energy_t,
        energy_x_t,
        energy_y_t,
        energy_t_ana,
        energy_x_t_ana,
        energy_y_t_ana,
        ts,
    )


def run_jimenez_1990(start_time=0):
    start_time = int(start_time)
    Re = 5000
    alpha = 1

    Nx = 100
    Ny = 140
    Nz = 2
    end_time = 1000

    nse = solve_navier_stokes_pertubation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(2 * jnp.pi / alpha, 1.0, 0.1),
    )

    nse.set_linearize(False)

    if start_time == 0:
        lsc = LinearStabilityCalculation(Re, alpha, Ny)
        vel_pert = lsc.velocity_field(nse.domain_no_hat)
        vort_pert = vel_pert.curl()
        # eps = 1e0 / jnp.sqrt(vort_pert.energy())
        eps = 1e-2 / jnp.sqrt(vel_pert.energy())
        nse.init_velocity((vel_pert * eps).hat())
    else:
        u = Field.FromFile(
            nse.domain_no_hat,
            "velocity_pertubation_" + str(0) + "_t_" + str(start_time),
            name="u_hat",
        )
        v = Field.FromFile(
            nse.domain_no_hat,
            "velocity_pertubation_" + str(1) + "_t_" + str(start_time),
            name="v_hat",
        )
        w = Field.FromFile(
            nse.domain_no_hat,
            "velocity_pertubation_" + str(2) + "_t_" + str(start_time),
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
                    "velocity_pertubation_0_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            [
                f.unlink()
                for f in Path("./fields/").glob(
                    "velocity_pertubation_1_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            [
                f.unlink()
                for f in Path("./fields/").glob(
                    "velocity_pertubation_2_t_" + str(i - 10)
                )
                if f.is_file()
            ]
            for j in range(3):
                vel_pert[j].save_to_file(
                    "velocity_pertubation_" + str(j) + "_t_" + str(i)
                )
                vort[j].plot_3d()
                vort[j].plot_3d(2)
                vort[j].plot_isolines(2)
                # vel_moving_frame[j].plot_3d(2)

    nse.before_time_step_fn = before_time_step
    nse.solve()


def run_transient_growth(Re=3000.0, T=15.0):
    Re = float(Re)
    T = float(T)
    alpha = 1

    eps = 1e-0

    number_of_modes = 80

    Nx = 40
    Ny = 100
    Nz = 4
    end_time = 1.1 * T

    lsc = LinearStabilityCalculation(Re, alpha, 100)

    nse = solve_navier_stokes_pertubation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        pertubation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi),
    )

    nse.set_linearize(True)

    U = lsc.calculate_transient_growth_initial_condition(
        nse.domain_no_hat,
        T,
        number_of_modes,
        recompute_full=False,
        recompute_partial=False,
        save_modes=False,
        save_final=True,
    )

    U.plot_streamlines(2)
    raise Exception("break")


    U_hat = U.hat()
    eps_ = eps / jnp.sqrt(U.energy())
    print("U energy norm: ", jnp.sqrt(U.energy()))
    print("U energy norm (RH): ", jnp.sqrt(U.energy_norm(1)))

    nse.init_velocity(U_hat * eps_)

    e_max = lsc.calculate_transient_growth_max_energy(
        nse.domain_no_hat, T, number_of_modes
    )
    print("expecting max growth of ", e_max)

    plot_interval = 50

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    energy_x_t = []
    energy_y_t = []
    energy_t_norm = []
    energy_x_t_norm = []
    energy_y_t_norm = []
    rh_93_data = genfromtxt(
        "rh93_transient_growth.csv", delimiter=","
    ).T  # TODO get rid of this at some point
    # energy_max = []
    energy_0_norm = vel_pert.energy_norm(1)
    print("inital pertubation energy norm: ", energy_0_norm)
    energy_0 = vel_pert.energy()
    print("inital pertubation energy: ", energy_0)

    def before_time_step(nse):
        i = nse.time_step
        if i % plot_interval == 0:
            vel_hat = nse.get_field("velocity_hat", i)
            vel = vel_hat.no_hat()
            vel_pert = VectorField([vel[0], vel[1], vel[2]])
            vel_pert_old = nse.get_field("velocity_hat", max(0, i - 1)).no_hat()
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
            vel_pert_energy_old = vel_pert_old.energy()
            vel_pert_energy_norm = vel_pert.energy_norm(1)
            print("\n\n")
            print(
                "velocity pertubation energy change: ",
                vel_pert_energy - vel_pert_energy_old,
            )
            print(
                "velocity pertubation energy x change: ",
                vel_pert[0].energy() - vel_pert_old[0].energy(),
            )
            print(
                "velocity pertubation energy y change: ",
                vel_pert[1].energy() - vel_pert_old[1].energy(),
            )
            print(
                "velocity pertubation energy z change: ",
                vel_pert[2].energy() - vel_pert_old[2].energy(),
            )
            print("")
            ts.append(nse.time)
            energy_t.append(vel_pert_energy / energy_0)
            energy_x_t.append(vel_pert[0].energy() / energy_0)
            energy_y_t.append(vel_pert[1].energy() / energy_0)
            energy_t_norm.append(vel_pert_energy_norm / energy_0_norm)
            energy_x_t_norm.append(vel_pert[0].energy_norm(1) / energy_0_norm)
            energy_y_t_norm.append(vel_pert[1].energy_norm(1) / energy_0_norm)
            # energy_max.append(lsc.calculate_transient_growth_max_energy(nse.domain_no_hat, nse.time, number_of_modes))

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            ax.plot(ts, energy_t, ".", label="growth (DNS)")
            ax.plot(ts, energy_t_norm, ".", label="growth of energy norm (DNS)")
            # ax.plot(ts, energy_max, ".", label="max growth (theory)")
            ax.plot(
                rh_93_data[0],
                rh_93_data[1],
                "--",
                label="growth (Reddy/Henningson 1993)",
            )
            ax.set_xlim([0, nse.time * 1.2])
            ax.set_ylim([0, e_max * 1.2])
            fig.legend()
            fig.savefig("plots/energy_t.pdf")

            fig_x = figure.Figure()
            ax_x = fig_x.subplots(1, 1)
            ax_x.plot(ts, energy_x_t, ".", label="growth")
            ax_x.plot(ts, energy_x_t_norm, ".", label="growth (norm)")
            fig_x.legend()
            fig_x.savefig("plots/energy_x_t.pdf")

            fig_y = figure.Figure()
            ax_y = fig_y.subplots(1, 1)
            ax_y.plot(ts, energy_y_t, ".", label="growth")
            ax_x.plot(ts, energy_y_t_norm, ".", label="growth (norm)")
            fig_y.legend()
            fig_y.savefig("plots/energy_y_t.pdf")

    nse.before_time_step_fn = before_time_step
    nse.after_time_step_fn = None

    nse.solve()


def run_optimization_pseudo_2d_pertubation():
    Re = 3000
    T = 1.0
    alpha = 1.02056
    Nx = 100
    Ny = 100
    Nz = 40
    # Nx = 20
    # Ny = 40
    # Nz = 10
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi)
    # Field.supress_plotting()

    def run(v0):
        (
            energy_t,
            energy_x_t,
            energy_y_t,
            energy_t_ana,
            energy_x_t_ana,
            energy_y_t_ana,
            ts,
        ) = run_pseudo_2d_pertubation(
            Re=Re,
            alpha=alpha,
            end_time=T,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            linearize=False,
            plot=False,
            save=False,
            v0=v0,
        )
        return energy_t[-1] / energy_t[0]
        # print("energy[0]")
        # print(energy_t[0])
        # print("energy[-1]")
        # print(energy_t[-1])
        # return energy_t[-1]

    Equation.initialize()


    dom = Domain((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)
    lsc = LinearStabilityCalculation(Re, alpha, Ny)
    v0_0 = lsc.velocity_field(dom, 0)


    v0s = [[v0_0[i].field for i in range(3)]]
    step_size = 1e-0
    sq_grad_sums = 0.0 * v0_0[0].field
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_arr = jnp.array(corr)
        corr_field = VectorField([Field(v0_0.domain, corr[i], name="correction_" + "xyz"[i]) for i in range(3)])
        corr_field.plot_3d(2)
        # corr_field.update_boundary_conditions()
        print("gain: " + str(gain))
        # print("corr (abs): " + str(abs(corr_field)))
        sq_grad_sums += corr_arr**2.0
        # alpha = jnp.array([eps / (1e-10 + jnp.sqrt(sq_grad_sums[i])) for i in range(v0_0[0].field.shape)])
        # eps = step_size / ((1 + 1e-10) * jnp.sqrt(sq_grad_sums))
        eps = step_size

        # print("eps")
        # print(eps)
        # print("sq_grad_sums")
        # print(sq_grad_sums)
        v0s.append([v0s[-1][j] + eps * corr_arr[j] for j in range(3)])
        v0_new = VectorField([Field(v0_0.domain, v0s[-1][j]) for j in range(3)])
        v0_new.set_name("vel_0_" + str(i))
        v0_new.plot_3d(2)

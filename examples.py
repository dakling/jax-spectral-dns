#!/usr/bin/env python3

import jax
# import jax.scipy.optimize as jaxopt
# import jaxopt
# import optax
import scipy.optimize as sciopt
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import matplotlib.figure as figure
from functools import partial
import typing
import time

# from importlib import reload
import sys

from cheb import cheb
# try:
#     reload(sys.modules["domain"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load Domain")
from domain import PhysicalDomain

# try:
#     reload(sys.modules["field"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load Field")
from field import FourierField, PhysicalField, FourierFieldSlice, VectorField

# try:
#     reload(sys.modules["equation"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load Equation")
from equation import Equation

# try:
#     reload(sys.modules["navier_stokes"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load Navier Stokes")
from navier_stokes import NavierStokesVelVort, solve_navier_stokes_laminar

# try:
#     reload(sys.modules["navier_stokes_perturbation"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load navier-stokes-perturbation")
from navier_stokes_perturbation import NavierStokesVelVortPerturbation, solve_navier_stokes_perturbation

# try:
#     reload(sys.modules["linear_stability_calculation"])
# except:
#     if hasattr(sys, "ps1"):
#         print("Unable to load linear stability")
from linear_stability_calculation import LinearStabilityCalculation

NoneType = type(None)

def run_optimization():
    Re = 1e0
    Ny = 24
    end_time = 1

    nse = solve_navier_stokes_laminar(
        Re=Re, Ny=Ny, end_time=end_time, perturbation_factor=0.0
    )

    def run(v0):
        nse_ = solve_navier_stokes_laminar(
            Re=Re, Ny=Ny, end_time=end_time, max_iter=10, perturbation_factor=0.0
        )
        nse_.max_iter = 10
        v0_field = PhysicalField(nse_.get_physical_domain()
, v0)
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
    v0_0 = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_x_fn_ana)

    v0s = [v0_0.data]
    eps = 1e3
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_field = PhysicalField(nse.get_physical_domain()
, corr, name="correction")
        corr_field.update_boundary_conditions()
        print("gain: " + str(gain))
        print("corr (abs): " + str(abs(corr_field)))
        v0s.append(v0s[-1] + eps * corr_field.data)
        v0_new = PhysicalField(nse.get_physical_domain()
, v0s[-1])
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
        perturbation_factor=0.1
        # Re=Re, Ny=12, Nx=4, end_time=end_time, perturbation_factor=1
        # Re=Re, Ny=48, Nx=24, end_time=end_time, perturbation_factor=1
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
    vel_x = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_x_fn, name="velocity_x")

    # vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) * vortex_sum(X)[1]
    vel_y_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + 0.1 * jnp.sin(
        2 * jnp.pi * X[2] / Lz * omega
    )
    # vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2] + (1 - X[1]**2) *vortex_sum(X)[2]
    vel_z_fn = lambda X: 0.0 * X[0] * X[1] * X[2]
    vel_y = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_y_fn, name="velocity_y")
    vel_z = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_z_fn, name="velocity_z")
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
            vel = nse.get_latest_field("velocity_hat").no_hat
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

    energy_over_time_fn_raw, ev = lsc.energy_over_time(nse.get_physical_domain()
)
    energy_over_time_fn = lambda t: eps**2 * energy_over_time_fn_raw(t)
    energy_x_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.get_physical_domain()

    )[0](t, 0)
    energy_y_over_time_fn = lambda t: eps**2 * lsc.energy_over_time(
        nse.get_physical_domain()

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
            # vel_hat = nse.get_field("velocity_hat", i)
            vel_hat = nse.get_latest_field("velocity_hat")
            vel = vel_hat.no_hat()
            vel_x_max = vel[0].max()
            print("vel_x_max: ", vel_x_max)
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
            print(
                "analytical velocity perturbation energy: ",
                energy_over_time_fn(nse.time),
            )
            print("velocity perturbation energy: ", vel_pert_energy)
            print("velocity perturbation energy x: ", vel_pert[0].energy())
            print(
                "analytical velocity perturbation energy x: ",
                energy_x_over_time_fn(nse.time),
            )
            print("velocity perturbation energy y: ", vel_pert[1].energy())
            print(
                "analytical velocity perturbation energy y: ",
                energy_y_over_time_fn(nse.time),
            )
            print("velocity perturbation energy z: ", vel_pert[1].energy())
            # vel_pert_energy_old = vel_pert_old.energy()
            # print(
            #     "velocity perturbation energy change: ",
            #     vel_pert_energy - vel_pert_energy_old,
            # )
            # print(
            #     "velocity perturbation energy x change: ",
            #     vel_pert[0].energy() - vel_pert_old[0].energy(),
            # )
            # print(
            #     "velocity perturbation energy y change: ",
            #     vel_pert[1].energy() - vel_pert_old[1].energy(),
            # )
            # print(
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
    vel_x = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_x_fn, name="velocity_x")
    vel_y = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_y_fn, name="velocity_y")
    vel_z = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_z_fn, name="velocity_z")
    nse.set_field(
        "velocity_hat", 0, VectorField([vel_x.hat(), vel_y.hat(), vel_z.hat()])
    )

    plot_interval = 1

    vel = nse.get_initial_field("velocity_hat").no_hat()
    vel_x_fn_ana = lambda X: -1 * (X[1] + 1) * (X[1] - 1) + 0.0 * X[0] * X[2]
    vel_x_ana = PhysicalField.FromFunc(nse.get_physical_domain()
, vel_x_fn_ana, name="vel_x_ana")

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
            print("velocity perturbation energy: ", vel_pert_energy)
            print("velocity perturbation abs: ", vel_pert_abs)

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
        jit=True
):
    Re = float(Re)
    alpha = float(alpha)
    end_time = float(end_time)
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
            rotated=False
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
            rotated=True
        )


    nse.set_linearize(linearize)
    # nse.initialize()

    if type(v0) == NoneType:
        U = lsc.velocity_field_single_mode(nse.get_physical_domain(), save=save)
    else:
        # U = VectorField([Field(nse.get_physical_domain(), v0[i]) for i in range(3)]).normalize()
        U = VectorField([PhysicalField(nse.get_physical_domain(), v0[i]) for i in range(3)])
        print(U[0].energy())


    if rotated:
        U_ = lsc.velocity_field_single_mode(PhysicalDomain.create((Nz, Ny, Nx), (True, False, True), scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 1e-3), aliasing=aliasing), save=save)
        U = VectorField([PhysicalField(nse.get_physical_domain(), jnp.moveaxis(jnp.moveaxis(U_[2].data, 0, 2), 0, 1)),
                         PhysicalField(nse.get_physical_domain(), jnp.moveaxis(jnp.moveaxis(U_[1].data, 0, 2), 0, 1)),
                         PhysicalField(nse.get_physical_domain(), jnp.moveaxis(jnp.moveaxis(U_[0].data, 0, 2), 0, 1))
                         ])

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
        for j in range(3):
            vel[j].time_step = i
            vort[j].time_step = i
            vel[j].name = "velocity_" + "xyz"[j]
            vort[j].name = "vorticity_" + "xyz"[j]
        if rotated:
            vel[2].plot_3d(0)
        else:
            vel[0].plot_3d(2)
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
    fig.savefig("plots/energy_t.pdf")

    return (
        energy_t,
        energy_x_t,
        energy_y_t,
        energy_t_ana,
        energy_x_t_ana,
        energy_y_t_ana,
        ts,
        vel_pert
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


def run_transient_growth(Re=3000.0, T=15.0, alpha=1.0, beta=0.0):

    # print("CPU?", jax.devices()[0].platform == "cpu")
    # print("GPU?", jax.devices()[0].platform == "gpu")
    # ensure that these variables are not strings as they might be passed as command line arguments
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    eps = 1e-5


    Nx = 10
    Ny = 90
    Nz = 6
    end_time = 1.01 * T
    number_of_modes = 80

    nse = solve_navier_stokes_perturbation(
        Re=Re,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        end_time=end_time,
        perturbation_factor=0.0,
        scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi),
        dt=1e-2,
        aliasing=1.0
    )
    nse.initialize()

    # nse.set_linearize(False)
    nse.set_linearize(True)

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)

    U = lsc.calculate_transient_growth_initial_condition(
        nse.get_physical_domain(),
        T,
        number_of_modes,
        recompute_full=True,
        save_final=True,
    )

    # U.plot_streamlines(2)
    # U[0].plot_3d(2)
    # U[1].plot_3d(2)

    U_hat = U.hat()
    eps_ = eps / jnp.sqrt(U.energy())
    print("U energy norm: ", jnp.sqrt(U.energy()))
    # print("U energy norm (RH): ", jnp.sqrt(U.energy_norm(1)))

    nse.init_velocity(U_hat * eps_)

    e_max = lsc.calculate_transient_growth_max_energy(
        nse.get_physical_domain(), T, number_of_modes
    )
    print("expecting max growth of ", e_max)

    vel_pert = nse.get_initial_field("velocity_hat").no_hat()
    vel_pert_0 = vel_pert[1]
    vel_pert_0.name = "veloctity_y_0"
    ts = []
    energy_t = []
    rh_93_data = np.genfromtxt(
        "rh93_transient_growth.csv", delimiter=","
    ).T  # TODO get rid of this at some point

    def post_process(nse, i):
        n_steps = len(nse.get_field("velocity_hat"))
        vel_hat = nse.get_field("velocity_hat", i)
        vel = vel_hat.no_hat()
        time = (i / (n_steps - 1)) * end_time

        vel.plot_streamlines(2)
        # vort = vel.curl()
        vel.set_time_step(i)
        vel.set_name("velocity")
        vel.plot_3d(2)
        vel[0].plot_center(0)
        vel[0].plot_center(1)
        vel_energy = vel.energy()
        ts.append(time)
        energy_t.append(vel_energy)
        # energy_x_t.append(vel_pert[0].energy() / energy_0)
        # energy_y_t.append(vel_pert[1].energy() / energy_0)
        # energy_t_norm.append(vel_pert_energy_norm / energy_0_norm)
        # energy_max.append(lsc.calculate_transient_growth_max_energy(nse.get_physical_domain(), nse.time, number_of_modes))


        # for i in range(3):
        #     vel[i].save_to_file(str(nse.time_step))

    # nse.before_time_step_fn = lambda nse: post_process(nse, nse.time_step)
    nse.before_time_step_fn = None
    nse.after_time_step_fn = None
    nse.post_process_fn = post_process

    nse.activate_jit()
    nse.write_intermediate_output = True
    nse.solve()
    nse.deactivate_jit()
    nse.post_process()

    energy_t = np.array(energy_t)
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    ax.plot(ts, energy_t/energy_t[0], ".", label="growth (DNS)")
    ax.plot(
        rh_93_data[0],
        rh_93_data[1],
        "--",
        label="growth (Reddy/Henningson 1993)",
    )
    # ax.set_xlim([0, end_time * 1.2])
    # ax.set_ylim([0, e_max * 1.2])
    fig.legend()
    fig.savefig("plots/energy_t.pdf")

    # fig_x = figure.Figure()
    # ax_x = fig_x.subplots(1, 1)
    # ax_x.plot(ts, energy_x_t, ".", label="growth")
    # fig_x.legend()
    # fig_x.savefig("plots/energy_x_t.pdf")

    # fig_y = figure.Figure()
    # ax_y = fig_y.subplots(1, 1)
    # ax_y.plot(ts, energy_y_t, ".", label="growth")
    # fig_y.legend()
    # fig_y.savefig("plots/energy_y_t.pdf")
    print("final energy gain:", energy_t[-1]/energy_t[0])
    print("expected final energy gain:", e_max)


def run_optimization_pseudo_2d_perturbation():
    Re = 3000.0
    T = 1.0
    alpha = 1.02056
    Nx = 64
    Ny = 90
    Nz = 64
    # Nx = 20
    # Ny = 40
    # Nz = 10
    scale_factors = (1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi)
    # PhysicalField.activate_jit()

    def run(v0):
        (
            energy_t,
            energy_x_t,
            energy_y_t,
            energy_t_ana,
            energy_x_t_ana,
            energy_y_t_ana,
            ts,
        ) = run_pseudo_2d_perturbation(
            # Re=Re,
            # alpha=alpha,
            # end_time=T,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            linearize=False,
            plot=False,
            save=False,
            v0=v0,
        )
        return energy_t[-1] / energy_t[0]

    Equation.initialize()


    dom = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)
    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, n=Ny)
    v0_0 = lsc.velocity_field_single_mode(dom, 0)


    v0s = [[v0_0[i].field for i in range(3)]]
    step_size = 1e-0
    sq_grad_sums = 0.0 * v0_0[0].field
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run)(v0s[-1])
        corr_arr = jnp.array(corr)
        # corr_field = VectorField([Field(v0_0.domain, corr[i], name="correction_" + "xyz"[i]) for i in range(3)])
        # corr_field.plot_3d(2)
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

n_iter = 0

def run_optimization_transient_growth(Re=3000.0, T=0.1, alpha=1.0, beta=0.0):
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    Equation.initialize()
    Nx = 2
    Ny = 50
    Nz = 2
    # Nx = 48
    # Ny = 64
    # Nz = 12
    end_time = T
    # number_of_modes = 100
    number_of_modes = 50
    scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    # HACK
    domain: PhysicalDomain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)


    v0_0 = lsc.calculate_transient_growth_initial_condition(
        domain,
        T,
        number_of_modes,
        # recompute_full=False,
        recompute_full=True,
        # recompute_partial=False,
        recompute_partial=True,
        save_modes=False,
        save_final=True,
    )
    # v0_0 = VectorField([PhysicalField.FromFunc(domain, lambda X: -0.1 * (1 - X[1]**2) + 0*X[2]),
    #                     PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                     PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                     ])
    eps = 1e-5
    eps_ = eps / v0_0.energy()
    v0_0_norm = v0_0 * eps_

    # @partial(jax.checkpoint, policy=jax.checkpoint_policies.checkpoint_dots)
    def run_case(v0):

        # v0_ = v0.reshape((3, Nx, Ny, Nz))
        # U = VectorField([PhysicalField(domain, v0_[i,...]) for i in range(3)])
        # domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors)
        U = VectorField([PhysicalField(domain, v0[i]) for i in range(3)])
        eps = 1e-5
        eps_ = eps / U.energy()
        U_norm = U * eps_
        U_norm.update_boundary_conditions() # TODO possible even enfore bcs for derivatives

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, Re)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None

        gain = vel.energy() / vel_0.energy()
        return gain
        # return -gain # (TODO would returning 1/gain lead to a better minimization problem?)

    # res = jaxopt.minimize(
    #     fun=run_case,
    #     x0=v0_0.get_fields().flatten(),
    #     # method='l-bfgs-experimental-do-not-rely-on-this',
    #     method='bfgs',
    #     tol=tol,
    #     options = {'maxiter': 2},
    #     )
    # adapted from Joe's code
    # maxiter = 1000
    # tol = 1e-10
    # solver = jaxopt.ScipyMinimize(
    #     fun=run_case,
    #     method='L-BFGS-B',
    #     tol=tol,
    #     # jit=True, # TODO
    #     jit=False,
    #     maxiter = maxiter,
    #     # callback=callback
    # )
    # params, state = solver.run(v0_0.get_fields().flatten())

    # def callback(intermediate_result=None):
    #     global n_iter
    #     vel_opt = VectorField([PhysicalField(domain, intermediate_result.x.reshape((3, Nx, Ny, Nz))[i,...], name="velocity_opt_" + "xyz"[i]) for i in range(3)])
    #     vel_opt.set_time_step(n_iter)
    #     vel_opt.plot_3d(2)
    #     n_iter += 1

    # res = sciopt.minimize(
    #     fun=jax.value_and_grad(run_case),
    #     x0=v0_0.get_fields().flatten(),
    #     jac=True,
    #     method='L-BFGS-B',
    #     # method='CG',
    #     tol=tol,
    #     callback=callback
    # )

    # vel_opt = VectorField([PhysicalField(domain, res.x.reshape((3, Nx, Ny, Nz))[i,...], name="velocity_opt_" + "xyz"[i]) for i in range(3)])
    # vel_opt.plot_3d(2)

    v0s = [[v0_0_norm[i].data for i in range(3)]]
    step_size = 1e-2
    for i in jnp.arange(10):
        gain, corr = jax.value_and_grad(run_case)(v0s[-1])
        corr_arr = jnp.array(corr)
        corr_arr = corr_arr / jnp.linalg.norm(corr_arr) * jnp.linalg.norm(jnp.array(v0s[-1]))
        print("gain: " + str(gain))
        eps = step_size

        # v0s.append([v0s[-1][j] + eps * corr_arr[j] for j in range(3)])
        v0s[-1] = [v0s[-1][j] + eps * corr_arr[j] for j in range(3)]
        v0_new = VectorField([PhysicalField(v0_0_norm[j].physical_domain, v0s[-1][j]) for j in range(3)])
        v0_new.set_name("vel_0_" + str(i))
        v0_new.plot_3d(2)
        v0_new.save_to_file("vel_0_" + str(i))


def run_optimization_transient_growth_coefficients(Re=3000.0, T=0.5, alpha=1.0, beta=0.0, file=None):
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    Equation.initialize()
    Nx = 8
    Ny = 80
    Nz = 4
    end_time = T
    number_of_modes = 120
    scale_factors=(1 * (2 * jnp.pi / alpha), 1.0, 2 * jnp.pi * 1e-3)
    aliasing = 3/2

    lsc = LinearStabilityCalculation(Re=Re, alpha=alpha, beta=beta, n=Ny)
    # HACK
    domain: PhysicalDomain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors, aliasing=aliasing)
    energy_gain_svd = None

    def coeffs_to_complex_coeffs(c):
        half_len = len(c) // 2
        return c[:half_len] + 1j*c[half_len:]

    def complex_coeffs_to_coeffs(c):
        return jnp.block([c.real, c.imag])

    S, coeffs_orig = lsc.calculate_transient_growth_svd(domain, T, number_of_modes, save=False, recompute=True)
    energy_gain_svd = S[0]**2
    print("excpected energy gain:", energy_gain_svd)
    if file is not None:
        coeff_array = np.load(file, allow_pickle=True)
        coeffs = coeffs_to_complex_coeffs(jnp.array(coeff_array.tolist()))
    else:
        coeffs = coeffs_orig

    correct_coeffs = coeffs_to_complex_coeffs(jnp.array([4.34764619e-14,1.32303681e-02,3.92520959e-11,-9.04709746e-05,
                                                         2.15886763e-04,-1.02942018e-02,-1.09454748e-13,7.30117704e-13,
                                                         1.38533034e-14,3.30576081e-11,-5.59880684e-01,9.35123909e-05,
                                                         9.14614269e-14,-1.36998039e-11,5.21419267e-14,-6.98347617e-01,
                                                         -7.36970972e-12,2.93050962e-01,3.27559129e-04,-1.34436440e-04,
                                                         -1.25241080e-14,-1.63571438e-01,1.73528871e-11,-2.25564252e-04,
                                                         -1.13870293e-13,-2.19919116e-01,3.50275103e-11,-1.70394364e-04,
                                                         -7.37516719e-14,5.31533821e-02,-3.00539152e-11,-1.87271304e-04,
                                                         -1.77364016e-13,-4.35910260e-02,2.05732669e-11,-9.99991818e-05,
                                                         2.43203358e-14,-2.06514224e-02,-4.04271862e-11,-2.48697837e-05,
                                                         -1.24532779e-13,-1.21789120e-02,-2.62106858e-11,1.30298805e-05,
                                                         5.35540369e-14,7.51231714e-05,-5.99142964e-12,1.01050849e-05,
                                                         -1.19756743e-14,3.40971109e-03,-9.37127307e-14,9.57604569e-01,
                                                         1.11957355e-12,-7.18632262e-05,5.81332190e-06,-2.55611639e-01,
                                                         -1.09486395e-13,-1.15004488e-11,3.77841975e-14,-1.85682978e-11,
                                                         7.14887335e-01,-3.69828770e-04,-3.92353412e-14,-7.86532109e-12,
                                                         2.91993706e-14,-1.10741047e+00,9.56204055e-12,-6.29518366e-01,
                                                         1.40514777e-04,4.50561969e-04,-7.34234334e-16,2.18187559e-01,
                                                         2.61059172e-12,3.39071399e-04,1.05807592e-14,-5.35355624e-02,
                                                         -2.48892717e-11,-2.48769788e-04,-1.66931866e-13,9.26838885e-02,
                                                         -4.56410041e-11,-4.06999954e-05,-9.07165396e-14,-1.28950935e-02,
                                                         -4.83973899e-11,-3.79294387e-05,1.70990967e-13,-1.11876117e-02,
                                                         -6.96492243e-12,4.73699810e-05,-1.10341594e-14,-3.85886935e-03,
                                                         4.51245247e-12,1.92912285e-05,-5.49754581e-14,-1.77457465e-03,
                                                         1.27154635e-11,2.61933727e-06,3.69624291e-14,-1.59204845e-03]))

    # raise Exception("break")

    def run_case(coeffs_):

        start_time = time.time()
        U = lsc.calculate_transient_growth_initial_condition_from_coefficients(
            domain,
            coeffs_,
            recompute=False
        )
        eps = 1e-5
        eps_ = eps / U.energy()
        U_norm = U * eps_
        # U_norm.update_boundary_conditions()

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, Re, physical_domain=domain)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        print("preparation took", time.time() - start_time, "seconds")
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None

        gain = vel.energy() / vel_0.energy()
        # return gain
        # print("gain:", gain)
        print("\n\n")
        jax.debug.print("gain: {}", gain)
        if energy_gain_svd is not None:
            print("expected gain:", energy_gain_svd)
        print("\n\n")
        return -gain # (TODO would returning 1/gain lead to a better minimization problem?)

    # coeffs_list = [coeffs]
    # step_size = 5e-1
    # print(coeffs_list[-1])
    # number_of_steps = 1000
    # for i in range(number_of_steps):
    #     start_time = time.time()
    #     gain, corr = jax.value_and_grad(run_case)(coeffs_list[-1])
    #     corr_arr = jnp.array(corr)
    #     print("gain: " + str(-gain))
    #     if energy_gain_svd is not None:
    #         print("expected gain:", energy_gain_svd)
    #     print("whole iteration took", time.time() - start_time, "seconds")
    #     eps = step_size

    #     coeffs_list[-1] = coeffs_list[-1] - eps * corr_arr
    #     print(coeffs_list[-1])
    #     # coeff_array = np.array(coeffs_list[-1].tolist())
    #     # coeff_array.dump(PhysicalField.field_dir + "coeffs_" + str(i))

    # learning_rate = 1e-1
    # solver = optax.adagrad(learning_rate=learning_rate) # minimizer
    # # solver = optax.adabelief(learning_rate=learning_rate) # minimizer
    # # solver = optax.adam(learning_rate=learning_rate) # minimizer
    # opt_state = solver.init(coeffs)
    # number_of_steps = 1000
    # print(coeffs)
    # for i in range(number_of_steps):
    #     gain, corr = jax.value_and_grad(run_case)(coeffs)
    #     print("gain: " + str(-gain))

    #     updates, opt_state = solver.update(corr, opt_state, coeffs)
    #     coeffs = optax.apply_updates(coeffs, updates)
    #     print("coeffs:", coeffs)
    #     print("gradient magnitudes:", jnp.linalg.norm(corr))
    #     coeff_array = np.array(coeffs.tolist())
    #     coeff_array.dump(PhysicalField.field_dir + "coeffs_" + str(i))

    import matplotlib.pyplot as plt

    def plot(coeffs_new):
        i_s = np.arange(number_of_modes)
        fig = figure.Figure()
        # ax = fig.subplots(1, 3)
        ax = fig.subplots(3, 1)
        ax[0].set_yscale("log")
        ax[1].set_yscale("symlog")
        ax[2].set_yscale("symlog")
        ax[0].plot(i_s, abs(coeffs_orig), "x", label="initial")
        ax[1].plot(i_s, (coeffs_orig.real), "x")
        ax[2].plot(i_s, (coeffs_orig.imag), "x")
        if abs(Re - 600) < 1e-3:
            ax[0].plot(i_s[:50], abs(correct_coeffs), "o", label="previous optimization")
            ax[1].plot(i_s[:50], (correct_coeffs.real), "o")
            ax[2].plot(i_s[:50], (correct_coeffs.imag), "o")
        if abs(Re - 3000) < 1e-3:
            rh_93_coeffs = np.genfromtxt(
                "rh93_coeffs.csv", delimiter=","
            ).T
            ax[0].plot(rh_93_coeffs[0]-1, abs(rh_93_coeffs[1]), "k.", label="rh93")
            fig2 = figure.Figure()
            ax2 = fig2.subplots(1, 1)
            ax2.set_yscale("log")
            ax2.plot(rh_93_coeffs[0]-1, abs(rh_93_coeffs[1]), "k.", label="rh93")

            cut_off = 1e-3
            coeffs_orig_filtered = coeffs_orig[abs(coeffs_orig) >= cut_off]
            coeffs_new_filtered = coeffs_new[abs(coeffs_new) >= cut_off]
            i_s_1 = np.arange(len(coeffs_orig_filtered))
            i_s_2 = np.arange(len(coeffs_new_filtered))
            ax2.plot(i_s_1, abs(coeffs_orig_filtered), "x", label="initial")
            ax2.plot(i_s_2, abs(coeffs_new_filtered), ".", label="current")
            fig2.legend()
            fig2.savefig("plots/coeff_plot_rh.pdf")
        ax[0].plot(i_s, abs(coeffs_new), ".", label="current")
        ax[1].plot(i_s, (coeffs_new.real), ".")
        ax[2].plot(i_s, (coeffs_new.imag), ".")
        fig.legend()
        fig.savefig("plots/coeff_plot.pdf")

    def callback(intermediate_result=None):
        global n_iter
        coeff_array.dump(PhysicalField.field_dir + "coeffs_" + str(n_iter))
        print(coeff_array)
        n_iter += 1
        coeffs_new = coeffs_to_complex_coeffs(np.array(intermediate_result.x.tolist()))
        plot(coeffs_new)

    tol = 1e-8

    print(coeffs)
    plot(coeffs)
    assert jnp.linalg.norm(coeffs - coeffs_to_complex_coeffs(complex_coeffs_to_coeffs(coeffs))) < 1e-30
    res = sciopt.minimize(
        fun=jax.value_and_grad(lambda c: run_case(coeffs_to_complex_coeffs(c))),
        x0=complex_coeffs_to_coeffs(coeffs),
        jac=True,
        # method='L-BFGS-B',
        method='CG',
        tol=tol,
        callback=callback
    )
    print(res)

def run_optimization_transient_growth_coefficients_memtest(Re=3000.0, T=0.5, alpha=1.0, beta=0.0, file=None):
    Re = float(Re)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)

    Equation.initialize()
    Nx = 64
    Ny = 90
    Nz = 64
    end_time = T
    scale_factors=(1.87, 1.0, 0.93)

    # HACK
    domain: PhysicalDomain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=scale_factors, aliasing=1)
    def run_case(coeffs_):

        start_time = time.time()
        U = VectorField([PhysicalField.FromFunc(domain, lambda X: 0.1 * (coeffs_[0] * jnp.sin(X[0]) + coeffs_[1] * (1 - X[1]**2) + coeffs_[2] * jnp.cos(X[2]))) for _ in range(3)])
        eps = 1e-5
        eps_ = eps / U.energy()
        U_norm = U * eps_
        # U_norm.update_boundary_conditions()

        nse = NavierStokesVelVortPerturbation.FromVelocityField(U_norm, Re, physical_domain=domain, dt=1e-2)
        nse.end_time = end_time

        # nse.set_linearize(False)
        nse.set_linearize(True)

        vel_0 = nse.get_initial_field("velocity_hat").no_hat()
        nse.activate_jit()
        print("preparation took", time.time() - start_time, "seconds")
        nse.solve()
        vel = nse.get_latest_field("velocity_hat").no_hat()

        nse.before_time_step_fn = None
        nse.after_time_step_fn = None

        gain = vel.energy() / vel_0.energy()
        # return gain
        # print("gain:", gain)
        jax.debug.print("gain: {}", gain)
        return -gain # (TODO would returning 1/gain lead to a better minimization problem?)

    coeffs = jnp.array([1.0,2.0,3.0])
    coeffs_list = [coeffs]
    step_size = 5e-1
    print(coeffs_list[-1])
    number_of_steps = 2
    for i in range(number_of_steps):
        start_time = time.time()
        gain, corr = jax.value_and_grad(run_case)(coeffs_list[-1])
        corr_arr = jnp.array(corr)
        print("gain: " + str(-gain))
        print("whole iteration took", time.time() - start_time, "seconds")
        eps = step_size

        coeffs_list[-1] = coeffs_list[-1] - eps * corr_arr
        print(coeffs_list[-1])



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
        nse.get_physical_domain(),
        T,
        number_of_modes,
        recompute_full=False,
        # recompute_full=True,
        # recompute_partial=False,
        recompute_partial=True,
        save_modes=False,
        save_final=True,
    )

    # u = PhysicalField.FromFunc(nse.physical_domain, (lambda X: (X[0] + X[1]**2 + X[2]**3)), name="vel_x")
    # v = PhysicalField.FromFunc(nse.physical_domain, (lambda X: 2 * (X[0] + X[1]**2 + X[2]**3)), name="vel_y")
    # w = PhysicalField.FromFunc(nse.physical_domain, (lambda X: 3 * (X[0] + X[1]**2 + X[2]**3)), name="vel_z")
    # # u = PhysicalField.FromFunc(nse.physical_domain, (lambda X: (1.0 + 0.0*X[0]*X[1]*X[2])), name="vel_x")
    # # v = PhysicalField.FromFunc(nse.physical_domain, (lambda X: 2 * (1.0 + 0.0*X[0]*X[1]*X[2])), name="vel_y")
    # # w = PhysicalField.FromFunc(nse.physical_domain, (lambda X: 3 * (1.0 + 0.0*X[0]*X[1]*X[2])), name="vel_z")
    # U = VectorField([u, v, w], name="vel")

    eps_ = eps / jnp.sqrt(U.energy())
    U_hat = U.hat() * eps_
    print("U energy norm: ", jnp.sqrt(U.energy()))
    # print("U energy norm (RH): ", jnp.sqrt(U.energy_norm(1)))

    nse.init_velocity(U_hat)

    U_ = U * eps_
    energy0 = U_.energy()
    print("U energy norm (normalized): ", energy0)
    # U_ = U
    for i in range(3):
        U_[i].name = "uvw"[i]
        # U_hat[i].name = "uvw"[i]
        U_[i].save_to_file("uvw"[i])
        # print(i, U_[i])
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
            print("\n\n")
            print(
                "velocity perturbation energy: ",
                vel_pert_energy,
            )
            print(
                "velocity perturbation relative change: ",
                vel_pert_energy / energy0,
            )
            print(
                "velocity perturbation energy change: ",
                vel_pert_energy - energy0,
            )
            # print(
            #     "velocity perturbation energy x change: ",
            #     vel_pert[0].energy() - vel_pert_old[0].energy(),
            # )
            # print(
            #     "velocity perturbation energy y change: ",
            #     vel_pert[1].energy() - vel_pert_old[1].energy(),
            # )
            # print(
            #     "velocity perturbation energy z change: ",
            #     vel_pert[2].energy() - vel_pert_old[2].energy(),
            # )

    nse.before_time_step_fn = before_time_step

    nse.solve()

    U_ = nse.get_latest_field("velocity_hat").no_hat()
    for i in range(3):
        U_[i].name = "uvw"[i]
        # U_hat[i].name = "uvw"[i]
        U_[i].save_to_file("uvw"[i] + "_final")
        # print(i, U_[i])
        # U_hat[i].save_to_file("uvw"[i])


def run_get_mean_profile(Re=2000):
    Nx = 64
    Ny = 90
    Nz = 64
    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=(1.87, 1.0, 0.93))
    avg_vel_coeffs = np.loadtxt("./profiles/Re_tau_180_90_small_channel.csv")
    def get_vel_field(domain, cheb_coeffs):
        U_mat = np.zeros((Ny, len(cheb_coeffs)))
        for i in range(Ny):
            for j in range(len(cheb_coeffs)):
                U_mat[i, j] = cheb(j, 0, domain.grid[1][i])
        U_y_slice = U_mat @ cheb_coeffs
        u_data = np.moveaxis(np.tile(np.tile(U_y_slice, reps=(Nz, 1)), reps=(Nx, 1, 1)), 1, 2)
        vel_base = VectorField([PhysicalField(domain, jnp.asarray(u_data)),
                                PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
                                PhysicalField.FromFunc(domain, lambda X: 0*X[2])])
        return vel_base, U_y_slice

    vel_base, _ = get_vel_field(domain, avg_vel_coeffs)
    vel_base.set_name("velocity_base")
    # vel_base.plot_3d(0)
    # vel_base.plot_3d(1)
    # vel_base.plot_3d(2)

    # vel_base_max = vel_base[0].max()
    # fit_fn = lambda y, k, n: vel_base_max * (((1 - y**2)**(1/n) + (1 - y**1.1)**(1/k)) / 2)
    # fit_fn_jac = jax.jacrev(fit_fn)
    # out = sciopt.curve_fit(lambda y, m, n: vel_base_max * (1 - y**m)**(1/n), domain.grid[1], U_y_slice, p0=(2.0, 1.0), maxfev=60000)
    # print(out)
    # vel_base_fit, _ = get_vel_field(domain, avg_vel_coeffs[:16])
    # vel_base_fit = VectorField([PhysicalField.FromFunc(domain, lambda X: 0*X[2] + fit_fn(X[1], out[0], out[1])),
    #                             PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                             PhysicalField.FromFunc(domain, lambda X: 0*X[2])])
    # vel_base_fit = VectorField([PhysicalField.FromFunc(domain, lambda X: 0*X[2] + fit_fn(X[1], 4.0, 3.0)),
    #                             PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                             PhysicalField.FromFunc(domain, lambda X: 0*X[2])])

    # vel_base[0].plot_center(1, vel_base_fit[0])

    # vel_hat = vel.hat()
    # vel_hat.set_name("velocity_hat")
    # nse = NavierStokesVelVort(vel_hat, Re=Re)
    # nse.activate_jit()
    # nse.write_intermediate_output = True
    # nse.end_time = 10
    # nse.initialize()
    # nse.solve()
    # nse.deactivate_jit()

    # # post-processing
    # mean_vel = VectorField([PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                    PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
    #                    PhysicalField.FromFunc(domain, lambda X: 0*X[2])])
    # n = len(nse.get_field("velocity_hat"))
    # for i, velocity_hat in enumerate(nse.get_field("velocity_hat")):
    #     velocity = velocity_hat.no_hat()
    #     mean_vel += velocity / n
    #     velocity.set_name("velocity")
    #     velocity.set_time_step(i)
    #     velocity[0].plot_3d(2)
    #     velocity[0].plot_center(1)

    #     mean_vel += velocity / n

    #     mean_vel_ = mean_vel * n / (i+1)
    #     mean_vel_.set_name("mean_velocity")
    #     mean_vel_.set_time_step(i)
    #     mean_vel_[0].plot_3d(2)
    #     mean_vel_[0].plot_center(1)


def run_ld_2020(turb=True, Re_tau=180):
    Re_tau = float(Re_tau)
    turb = str(turb) == 'True'
    Nx = 48
    Ny = 80
    Nz = 32
    aliasing = 3/2
    # aliasing = 1
    Nx = int(Nx * ((3/2) / aliasing))
    Nz = int(Nz * ((3/2) / aliasing))
    # end_time = 0.7 # in ld2020 units
    end_time = 0.02 # in ld2020 units
    domain = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=(1.87, 1.0, 0.93), aliasing=aliasing)
    avg_vel_coeffs = np.loadtxt("./profiles/Re_tau_180_90_small_channel.csv")
    def get_vel_field(domain, cheb_coeffs):
        U_mat = np.zeros((Ny, len(cheb_coeffs)))
        for i in range(Ny):
            for j in range(len(cheb_coeffs)):
                U_mat[i, j] = cheb(j, 0, domain.grid[1][i])
        U_y_slice = U_mat @ cheb_coeffs
        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = np.moveaxis(np.tile(np.tile(U_y_slice, reps=(nz, 1)), reps=(nx, 1, 1)), 1, 2)
        vel_base = VectorField([PhysicalField(domain, jnp.asarray(u_data)),
                                PhysicalField.FromFunc(domain, lambda X: 0*X[2]),
                                PhysicalField.FromFunc(domain, lambda X: 0*X[2])])
        return vel_base, U_y_slice

    if turb:
        print("using turbulent base profile")
        vel_base, _ = get_vel_field(domain, avg_vel_coeffs)
        vel_base, max = vel_base.normalize_by_max_value()
        vel_base.set_name("velocity_base")
        u_max_over_u_tau = max[0]
    else:
        print("using laminar base profile")
        vel_base = VectorField([PhysicalField.FromFunc(domain, lambda X: 1.0 * (1 - X[1]**2) + 0*X[2]),
                                    PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1]**2)  + 0*X[2]),
                                    PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1]**2)  + 0*X[2])])

        vel_base.set_name("velocity_base")
        u_max_over_u_tau = 18.5 # matches Vilda's profile

    vel_pert = VectorField([PhysicalField.FromFunc(domain, lambda X: 0.1 * (1 - X[1]**2) * 0.5 * (0*jnp.cos(1/0.5 * 2 * jnp.pi / 1.87 * X[0]) + 1*jnp.cos(1/0.5 * 2 * jnp.pi / 0.93 * X[2]))),
                            PhysicalField.FromFunc(domain, lambda X: 0.0 * (1 - X[1]**2) * 0.5 * (0.1*jnp.cos(1/0.5 * 2 * jnp.pi / 1.87 * X[0]) + 0.1*jnp.cos(1/0.5 * 2 * jnp.pi / 0.93 * X[2]))),
                            PhysicalField.FromFunc(domain, lambda X: 0.01 * (1 - X[1]**2) * 0.5 * (0.1*jnp.cos(1/0.5 * 2 * jnp.pi / 1.87 * X[0]) + 0.0*jnp.cos(1/0.5 * 2 * jnp.pi / 0.93 * X[2])))])
    vel_pert.set_name("velocity")

    vel_hat = vel_pert.hat()
    vel_hat.set_name("velocity_hat")

    Re = Re_tau * u_max_over_u_tau
    end_time_ = end_time * u_max_over_u_tau
    energy_0 = 1e-6

    def run_case(vel_data):
        domain_ = PhysicalDomain.create((Nx, Ny, Nz), (True, False, True), scale_factors=(1.87, 1.0, 0.93), aliasing=aliasing)
        vel_hat = VectorField([FourierField(domain_, vel_data[0]),
                               FourierField(domain_, vel_data[1]),
                               FourierField(domain_, vel_data[2])])
        vel_pert = vel_hat.no_hat()
        vel_pert.update_boundary_conditions()
        vel_pert.normalize_by_energy()
        vel_pert *= jnp.sqrt(energy_0)
        vel_hat = vel_pert.hat()
        vel_hat.set_name("velocity_hat")
        energy_0_ = vel_hat.no_hat().energy()
        jax.debug.print("initial energy: {x}", x=energy_0_)
        nse = NavierStokesVelVortPerturbation(vel_hat, Re_tau=Re, velocity_base_hat=vel_base.hat(), dt=1e-2)
        # jax.debug.print("recommended time step: {x}", x=nse.get_time_step())
        nse.activate_jit()
        # nse.write_intermediate_output = False
        nse.write_intermediate_output = True
        nse.end_time = end_time_
        # nse.initialize()
        nse.solve()
        vel_final = nse.get_latest_field("velocity_hat").no_hat()
        gain = vel_final.energy() / energy_0_
        jax.debug.print("gain: {x}", x=gain)
        return gain, nse

    v0s = [vel_hat.get_data()]
    # gain, nse = run_case(v0s[-1])
    # nse.initialize()
    # print("gain:", gain)
    # for i, vel_hat in enumerate(nse.get_field("velocity_hat")):
    #     vel = vel_hat.no_hat()
    #     vel.set_name("velocity")
    #     vel.set_time_step(i)
    #     vel.plot_3d(2)
    # raise Exception("done")
    step_size = 1e-2
    for i in jnp.arange(100):
        start_time = time.time()
        (gain, nse), corr = jax.value_and_grad(run_case, has_aux=True)(v0s[-1])
        corr_arr = jnp.array(corr)
        corr_arr = corr_arr / jnp.linalg.norm(corr_arr) * jnp.linalg.norm(jnp.array(v0s[-1]))
        print("gain: " + str(gain))
        eps = step_size

        # v0s.append([v0s[-1][j] + eps * corr_arr[j] for j in range(3)])
        v0s[-1] = jnp.array([v0s[-1][j] + eps * corr_arr[j] for j in range(3)])
        v0_new = VectorField([FourierField(domain, v0s[-1][j]).no_hat() for j in range(3)])
        v0_new.normalize_by_energy()
        v0_new *= energy_0
        v0_new.set_name("vel_0")
        v0_new.set_time_step(i)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1, FourierField(domain, v0s[0][0]).no_hat())
        v0_new.save_to_file("vel_0_" + str(i))
        corr = VectorField([FourierField(domain, corr[j]).no_hat() for j in range(3)])
        corr.set_name("corr")
        corr.set_time_step(i)
        corr.plot_3d(2)

    #     # nse.initialize()
    #     print("gain:", gain)
    #     # for j, vel_hat in enumerate(nse.get_field("velocity_hat")):
    #     #     vel = vel_hat.no_hat()
    #     #     vel.set_name("velocity_" + str(i))
    #     #     vel.set_time_step(j)
    #     #     vel.plot_3d(2)
    #     print("Iteration took", time.time() - start_time, "seconds")

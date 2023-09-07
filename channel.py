#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax_cfd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jax_cfd.base as cfd
from jax_cfd.base.grids import GridArray, GridVariable
import xarray
import pickle

def write_eps(eps):
    with open("eps.txt", "w+") as eps_file:
        eps_file.write('%d' % eps)

def read_eps():
    with open("eps.txt", "r") as eps_file:
        eps = float(eps_file.readlines()[0])
    return eps

def write_max_iter(max_iter):
    with open("max_iter.txt", "w+") as mi_file:
        mi_file.write('%d' % max_iter)

def read_max_iter():
    with open("max_iter.txt", "r") as mi_file:
        max_iter = int(mi_file.readlines()[0])
    return max_iter

def write_state(u0):
    with open("state", "wb") as f:
        pickle.dump(u0, f)

def read_state():
    with open("state", "rb") as f:
        u0 = pickle.load(f)
    return u0


def plot_state(v0, grid, ii):
    run_flow_sim_channel(v0, grid, ii)

def create_grid():
    size = (512, 128)
    domain = ((0, 8), (-1, 1))
    return cfd.grids.Grid(size, domain=domain)

## adapted from demo worksheet
def run_flow_sim_channel(v0, grid, ii=-1):

    density = 1.
    viscosity = 1e-3  # kinematic visocity
    pressure_gradient = 2e-3  # uniform dP/dx

    # Specify zero velocity initial conditions
    vx_fn = lambda x, y: jnp.zeros_like(x + y)
    vy_fn = lambda x, y: jnp.zeros_like(x + y)
    vz_fn = lambda x, y: jnp.zeros_like(x + y)

    # Specify a fixed time step based on the convection and diffusion scales
    max_velocity = 1  # value selected from known equilibirum profile
    cfl_safety_factor = 0.5

    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid)
    def pressure_gradient_forcing(pressure_gradient: float):

        def forcing(v):
            force_vector = (pressure_gradient, 0)
            return tuple(cfd.grids.GridArray(f * jnp.ones_like(u.data), u.offset, u.grid)
                        for f, u in zip(force_vector, v))

        return forcing

    # time steps per output
    inner_steps = 50

    # number of outputs
    outer_steps = 5

    # TODO(pnorgaard): Fix Van-Leer limiters for Dirichlet velocity BC
    def convect(v):
        return tuple(cfd.advection.advect_van_leer(u, v, dt) for u in v)

    # Define a step function and use it to compute a trajectory.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density,
            viscosity=viscosity,
            dt=dt,
            grid=grid,
            convect=convect,
            pressure_solve=cfd.pressure.solve_fast_diag_channel_flow,
            forcing=pressure_gradient_forcing(pressure_gradient),
            ),
        steps=inner_steps)
    rollout_fn = jax.jit(cfd.funcutils.trajectory(
        step_fn, outer_steps, start_with_input=True))

    _, trajectory = jax.device_get(rollout_fn(v0))

    def energy_field(ds):
        return (0.5*(ds.u**2 + ds.v**2)).rename('energy')


    def energy_at_time(time_index):
        x, y = grid.axes()
        u = trajectory[0].data
        v = trajectory[1].data
        data = jnp.array(0.5 * (u[time_index]**2 + v[time_index]**2))
        energy = jnp.trapz(jnp.trapz(data, x, axis=0), y, axis=0)
        return energy

    def gain():
        return energy_at_time(-1)/energy_at_time(0)

    # load into xarray for visualization and analysis
    if ii >= 0:
        ds = xarray.Dataset(
            {
                'u': (('time', 'x', 'y'), trajectory[0].data),
                'v': (('time', 'x', 'y'), trajectory[1].data),
            },
            coords={
                'x': grid.axes()[0],
                'y': grid.axes()[1],
                'time': dt * inner_steps * np.arange(outer_steps)
            }
        )

        plt = (ds.pipe(lambda ds: ds.u)#.thin(time=20)
        .plot.imshow(col='time', cmap=sns.cm.icefire, robust=True, col_wrap=5));
        plt.fig.savefig("plot_u_" + str(ii) + ".pdf")
        plt.fig.savefig("plot_u_" "latest" + ".pdf")
        plt = (ds.pipe(lambda ds: ds.v)#.thin(time=20)
        .plot.imshow(col='time', cmap=sns.cm.icefire, robust=True, col_wrap=5));
        plt.fig.savefig("plot_v_" + str(ii) + ".pdf")
        plt.fig.savefig("plot_v_" "latest" + ".pdf")
        plt = (ds.pipe(energy_field)#.thin(time=20)
        .plot.imshow(col='time', cmap=sns.cm.icefire, robust=True, col_wrap=5));
        plt.fig.savefig("plot_energy_" + str(ii) + ".pdf")
        plt.fig.savefig("plot_energy_" "latest" + ".pdf")

    return gain()

def energy(var):
    x, y = var[0].grid.axes()
    u = var[0].data
    v = var[1].data
    data = jnp.array(0.5 * (u**2 + v**2))
    energy = jnp.trapz(jnp.trapz(data, x, axis=0), y, axis=0)
    return energy

def linCombGridVars(var1, a1, var2=None, a2=0.0):
    dim = len(var1)
    if var2:
        assert len(var2) == dim
        return tuple(GridVariable(a1 * var1[d].array + a2 * var2[d].array, var1[d].bc) for d in range(dim))
    else:
        return tuple(GridVariable(a1 * var1[d].array, var1[d].bc) for d in range(dim))


def optimize_channel():

    # Define the physical dimensions of the simulation.
    grid = create_grid()

    # u0_unnormalized = cfd.initial_conditions.initial_velocity_field(
    #     velocity_fns=(vx_fn, vy_fn),
    #     grid=grid,
    #     velocity_bc=velocity_bc,
    #     pressure_solve=cfd.pressure.solve_fast_diag_channel_flow,
    #     iterations=5)

    velocity_bc = (cfd.boundaries.channel_flow_boundary_conditions(ndim=2),
               cfd.boundaries.channel_flow_boundary_conditions(ndim=2))
    # vx_fn = lambda x, y: jnp.sin(y * jnp.pi) * (1 + y) * (1 - y)
    # vy_fn = lambda x, y: 0
    vx_fn = lambda x, y: jnp.ones_like(x + y) * jnp.sin(y * jnp.pi) * (1 + y) * (1 - y)
    vy_fn = lambda x, y: jnp.ones_like(x + y) * 0

    u0_unnormalized = cfd.initial_conditions.initial_velocity_field(
        velocity_fns=(vx_fn, vy_fn),
        grid=grid,
        velocity_bc=velocity_bc,
        pressure_solve=cfd.pressure.solve_fast_diag_channel_flow,
        iterations=5)
    # Construct a random initial velocity. The `filtered_velocity_field` function
    # ensures that the initial velocity is divergence free and it filters out
    # high frequency fluctuations.

    E0 = 0.1
    e0 = energy(u0_unnormalized)

    u0 = linCombGridVars(u0_unnormalized, jnp.sqrt(E0/e0))

    u0_corr = cfd.initial_conditions.initial_velocity_field((lambda x, y :0, lambda x, y: 0), grid)

    gain_func = lambda v0, ii: run_flow_sim_channel(v0, grid, ii)

    eps = 500.0 # TODO introduce more sophisticated adaptive eps
    write_eps(eps)
    u0_vec = [u0]
    write_state(u0_vec)

    old_gain = None
    i = 0
    strikes = 0
    max_iter = 200
    # max_iter = 2
    # plot_interval = None
    write_max_iter(max_iter)

    plot_interval = 1

    plot_state(u0_vec[-1], grid, 0)
    while strikes < 6 and i < max_iter:
        print("iteration: ", i)
        gain, u0_corr = jax.value_and_grad(gain_func, argnums=0)(u0_vec[-1], -1)
        print("initial energy:")
        print(energy(u0_vec[-1]))
        print("gain:")
        print(gain)
        if old_gain and old_gain > gain:
            print("decrease in gain detected; decreasing step size and redoing iteration.")
            eps *= 0.7
            write_eps(eps)
            # strikes += 1
            gain = old_gain
        else:
            if plot_interval and i % plot_interval == 0:
                plot_state(u0_vec[-1], grid, i)
            if old_gain and jnp.abs(old_gain - gain) < 1e-5:
                pass
                # print("no significant improvement in gain detected; if this continues, I am stopping the iteration.")
                # strikes += 1
            elif old_gain and gain/old_gain < 1.1:
                pass
                # print("only slow improvement in gain detected; increasing step size.")
                # eps *= 1.1
            else:
                pass
                # strikes = 0
            u0_new_unnormalized = linCombGridVars(u0_vec[-1], 1.0, u0_corr, eps)
            e0 = energy(u0_new_unnormalized)
            u0_vec.append(linCombGridVars(u0_new_unnormalized, jnp.sqrt(E0/e0)))
            write_state(u0_vec)
            # eps *= 1.2
            old_gain = gain
        i += 1
        max_iter = read_max_iter()
        eps = read_eps()

    # post-processing
    gain_func(u0_vec[0], 0)
    gain_func(u0_vec[-1], i)

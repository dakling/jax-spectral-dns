#!/usr/bin/env python3

import jax
import jax.scipy as jsp
import jax.numpy as jnp
import jax_cfd
import jax_cfd.base as cfd
from jax_cfd.base.grids import GridVariable
import numpy as np
import seaborn
import xarray

## adapted from demo worksheet
def run_flow_sim(v0, grid, ii=-1):
    density = 1.
    viscosity = 1e-3
    inner_steps = 25
    outer_steps = 200

    max_velocity = 2.0
    cfl_safety_factor = 0.5

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid)

    # Define a step function and use it to compute a trajectory.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid),
        steps=inner_steps)
    rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
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

        plt = (ds.pipe(lambda ds: ds.u).thin(time=20)
        .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
        plt.fig.savefig("plot_u_" + str(ii) + ".pdf")
        plt.fig.savefig("plot_u_" "latest" + ".pdf")
        plt = (ds.pipe(lambda ds: ds.v).thin(time=20)
        .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
        plt.fig.savefig("plot_v_" + str(ii) + ".pdf")
        plt.fig.savefig("plot_v_" "latest" + ".pdf")
        plt = (ds.pipe(energy_field).thin(time=20)
        .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
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
    # TODO generalize dimensions
    if var2:
        return (GridVariable(a1 * var1[0].array + a2 * var2[0].array, var1[0].bc), GridVariable(a1 * var1[1].array + a2 * var2[1].array, var1[1].bc))
    else:
        return (GridVariable(a1 * var1[0].array, var1[0].bc), GridVariable(a1 * var1[1].array, var1[1].bc))


def main():

    seed = 0

    max_velocity = 2.0

    size = 50

    # Define the physical dimensions of the simulation.
    grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Construct a random initial velocity. The `filtered_velocity_field` function
    # ensures that the initial velocity is divergence free and it filters out
    # high frequency fluctuations.
    u0_unnormalized = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, max_velocity)

    E0 = 0.1
    e0 = energy(u0_unnormalized)

    u0 = linCombGridVars(u0_unnormalized, jnp.sqrt(E0/e0))

    u0_corr = cfd.initial_conditions.initial_velocity_field((lambda x, y :0, lambda x, y: 0), grid)

    gain_func = lambda v0: run_flow_sim(v0, grid)
    gain_func_with_postprocessing = lambda v0, ii: run_flow_sim(v0, grid, ii)

    eps = 10.0 # TODO introduce more sophisticated adaptive eps
    u0_vec = [u0]
    old_gain = None
    i = 0
    strikes = 0
    while strikes < 2:
        print("iteration: ", i)
        gain = gain_func_with_postprocessing(u0_vec[-1], i)
        print("initial energy:")
        print(energy(u0_vec[-1]))
        print("gain:")
        print(gain)
        if old_gain and old_gain > gain:
            print("decrease in gain detected; decreasing step size and redoing iteration.")
            eps *= 0.7
            strikes += 1
        else:
            if old_gain and  jnp.abs(old_gain - gain) < 1e-5:
                print("no significant improvement in gain detected; if this continues, I am stopping the iteration.")
                strikes += 1
            else:
                strikes = 0
            u0_corr = jax.grad(gain_func)(u0_vec[-1])
            u0_new_unnormalized = linCombGridVars(u0_vec[-1], 1.0, u0_corr, eps)
            e0 = energy(u0_new_unnormalized)
            u0_vec.append(linCombGridVars(u0_new_unnormalized, jnp.sqrt(E0/e0)))
            # eps *= 1.2
        old_gain = gain
        i += 1

    ds = xarray.Dataset(
        {
            'u0_0': (('x', 'y'), u0_vec[0][0].data),
            'v0_0': (('x', 'y'), u0_vec[0][1].data),
            'u0_last': (('x', 'y'), u0_vec[-1][0].data),
            'v0_last': (('x', 'y'), u0_vec[-1][1].data),
        },
        coords={
            'x': grid.axes()[0],
            'y': grid.axes()[1],
            'time': 0
        }
    )
    plt = (ds.pipe(lambda ds: ds.u0_last)
    .plot.imshow(cmap=seaborn.cm.icefire, robust=True, col_wrap=5)).get_figure()
    plt.clf()
    plt = (ds.pipe(lambda ds: ds.u0_0)
    .plot.imshow(cmap=seaborn.cm.icefire, robust=True, col_wrap=5)).get_figure()
    plt.savefig("plot_0.pdf")
    plt.clf()
    plt = (ds.pipe(lambda ds: ds.u0_last)
    .plot.imshow(cmap=seaborn.cm.icefire, robust=True, col_wrap=5)).get_figure()
    plt.savefig("plot_last.pdf")

main()

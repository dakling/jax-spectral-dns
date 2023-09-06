#!/usr/bin/env python3

import jax
import jax.scipy as jsp
import jax.numpy as jnp
import jax_cfd
import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
from jax_cfd.spectral import utils as spectral_utils
from jax_cfd.base.grids import GridVariable
import numpy as np
import seaborn as sns
import xarray


## adapted from demo worksheet
def run_flow_sim_spectral(v0, grid, ii=-1):
    density = 1.
    viscosity = 1e-3
    max_velocity = 2.0
    cfl_safety_factor = 0.5

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid)

    final_time = 25.0
    outer_steps = 10
    inner_steps = (final_time // dt) // 10


    # Define a step function and use it to compute a trajectory.
    # step_fn = cfd.funcutils.repeated(
    #     cfd.equations.semi_implicit_navier_stokes(
    #         density=density, viscosity=viscosity, dt=dt, grid=grid),
    #     steps=inner_steps)
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(viscosity, grid, smooth=True), dt)

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    def vel_field(da, time_index):
        x, y = grid.axes()
        uhat, vhat = spectral_utils.vorticity_to_velocity(grid)(da[time_index])
        u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)
        return (u, v)

    def energy_field(da, time_index):
        u, v = vel_field(da, time_index)
        return (0.5*(u**2 + v**2))

    def energy_at_time(time_index):
        x, y = grid.axes()
        uhat, vhat = spectral_utils.vorticity_to_velocity(grid)(trajectory[time_index])
        u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)
        energy_field = 0.5 * (u**2 + v**2)
        energy = jnp.trapz(jnp.trapz(energy_field, x, axis=0), y, axis=0)
        return energy

    def gain():
        return energy_at_time(-1)/energy_at_time(0)

    # load into xarray for visualization and analysis
    if ii >= 0:
    # if True:
        # ds = xarray.Dataset(
        #     {
        #         'u': (('time', 'x', 'y'), trajectory[0].data),
        #         'v': (('time', 'x', 'y'), trajectory[1].data),
        #     },
        #     coords={
        #         'x': grid.axes()[0],
        #         'y': grid.axes()[1],
        #         'time': dt * inner_steps * np.arange(outer_steps)
        #     }
        # )

        # plt = (ds.pipe(lambda ds: ds.u).thin(time=20)
        # .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
        # plt.fig.savefig("plot_u_" + str(ii) + ".pdf")
        # plt.fig.savefig("plot_u_" "latest" + ".pdf")
        # plt = (ds.pipe(lambda ds: ds.v).thin(time=20)
        # .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
        # plt.fig.savefig("plot_v_" + str(ii) + ".pdf")
        # plt.fig.savefig("plot_v_" "latest" + ".pdf")
        # plt = (ds.pipe(energy_field).thin(time=20)
        # .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5));
        # plt.fig.savefig("plot_energy_" + str(ii) + ".pdf")
        # plt.fig.savefig("plot_energy_" "latest" + ".pdf")

        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
        coords = {
            'time': dt * jnp.arange(outer_steps) * inner_steps,
            'x': spatial_coord,
            'y': spatial_coord,
        }

        da = xarray.DataArray(jnp.fft.irfftn(trajectory, axes=(1,2)), dims=["time", "x", "y"], coords=coords)
        plt = da.plot.imshow(col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True).fig
        plt.savefig("plot_vort_" + str(ii) + ".pdf")
        plt.savefig("plot_vort_" "latest" + ".pdf")

    return gain()

# def energy_at_time(time_index):
#     x, y = grid.axes()
#     uhat, vhat = spectral_utils.vorticity_to_velocity(grid)(trajectory[time_index])
#     u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)
#     energy_field = 0.5 * (u**2 + v**2)
#     energy = jnp.trapz(jnp.trapz(energy_field, x, axis=0), y, axis=0)
#     return energy
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


def optimize_spectral():

    seed = 0

    max_velocity = 2.0

    size = 256

    # Define the physical dimensions of the simulation.
    grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Construct a random initial velocity. The `filtered_velocity_field` function
    # ensures that the initial velocity is divergence free and it filters out
    # high frequency fluctuations.
    u0_unnormalized = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
    u0 = u0_unnormalized

    E0 = 0.1
    e0 = energy(u0_unnormalized)
    # e0 = 0.1

    u0 = linCombGridVars(u0_unnormalized, jnp.sqrt(E0/e0))

    u0_corr = cfd.initial_conditions.initial_velocity_field((lambda x, y :0, lambda x, y: 0), grid)

    gain_func = lambda v0: run_flow_sim_spectral(v0, grid)
    gain_func_with_postprocessing = lambda v0, ii: run_flow_sim_spectral(v0, grid, ii)

    eps = 100.0 # TODO introduce more sophisticated adaptive eps
    u0_vec = [u0]
    old_gain = None
    i = 0
    strikes = 0
    while strikes < 2 and i < 80:
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
            if old_gain and jnp.abs(old_gain - gain) < 1e-5:
                print("no significant improvement in gain detected; if this continues, I am stopping the iteration.")
                strikes += 1
            elif old_gain and old_gain/gain < 1.2:
                print("only slow improvement in gain detected; increasing step size.")
            eps *= 1.5
            else:
                strikes = 0
            u0_corr = jax.grad(gain_func)(u0_vec[-1])
            u0_new_unnormalized = linCombGridVars(u0_vec[-1], 1.0, u0_corr, eps)
            e0 = energy(u0_new_unnormalized)
            u0_vec.append(linCombGridVars(u0_new_unnormalized, jnp.sqrt(E0/e0)))
            # eps *= 1.2
        old_gain = gain
        i += 1

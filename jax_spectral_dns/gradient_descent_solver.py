#!/usr/bin/env python

from abc import ABC, abstractmethod
import os, glob
from shutil import copyfile
import time
import math
from typing import Any, Optional, Tuple, cast, TYPE_CHECKING
import jax
import jax.numpy as jnp
import jaxopt  # type: ignore[import-untyped]
from matplotlib import figure
import numpy as np
from matplotlib.axes import Axes, subplot_class_factory
from jax_spectral_dns.equation import Equation, print_verb
from jax_spectral_dns.field import Field, FourierField, VectorField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
from jax_spectral_dns.navier_stokes_perturbation_dual import (
    NavierStokesVelVortPerturbationDual,
)
from jax_spectral_dns.optimiser import OptimiserFourier

try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    from jax_spectral_dns._typing import jsd_float, parameter_type, jnp_array, jsd_array


class GradientDescentSolver(ABC):

    def __init__(
        self, dual_problem: NavierStokesVelVortPerturbationDual, **params: Any
    ):
        # initialisation
        self.dual_problem = dual_problem
        v0_no_hat = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        ).no_hat()
        self._e_0 = v0_no_hat.energy()
        self.done = False
        self.post_process_fn = params.get("post_process_function", None)
        self.value: Optional[float] = None
        self.old_value: Optional[float] = self.value

        # set various solver options
        self.i = params.get("start_iteration", 0)
        if self.i == -1:
            self.determine_last_iteration_step()
        self.trajectory_write_interval = params.get("trajectory_write_interval", 20)
        self.max_step_size = params.get("max_step_size", 1e-1)
        self.min_step_size = params.get("min_step_size", 1e-4)
        self.step_size = params.get("step_size", 1e-2)
        self.number_of_steps = params.get("max_iterations", 20)
        self.relative_gain_increase_threshold = params.get(
            "relative_gain_increase_threshold", 0.9
        )
        self.max_number_of_sub_iterations = params.get(
            "max_number_of_sub_iterations", 10
        )
        self.value_change_threshold = params.get("value_change_threshold", 1e-8)
        self.step_size_threshold = params.get("step_size_threshold", 1e-5)

        self.use_linesearch = params.get("use_linesearch", False)

        self.current_guess = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        )

    @property
    def e_0(self) -> float:
        return self._e_0

    def increase_step_size(self, tau: float = 1.5) -> None:
        self.step_size = min(self.max_step_size, self.step_size * tau)

    def decrease_step_size(self, tau: float = 5.0) -> None:
        self.step_size /= tau

    @abstractmethod
    def initialise(self, prepare_for_iterations: bool = True) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    def is_done(self, i: int) -> bool:
        done: bool = (
            (i >= self.number_of_steps)
            or (self.step_size < self.step_size_threshold)
            or (
                ((self.value is not None) and (self.old_value is not None))
                and (
                    abs((self.value - self.old_value) / self.value)
                    < self.value_change_threshold
                )
            )
        )
        return done

    def update_done(self, i: int) -> None:
        if self.is_done(i):
            self.done = True

    def optimise(self) -> None:
        self.initialise(self.number_of_steps >= 0)
        self.update_done(self.i)
        while not self.done:
            jax.clear_caches()  # type: ignore[no-untyped-call]
            self.update()
            self.post_process_iteration()
            self.i += 1
            self.update_done(self.i)
            assert self.value is not None
            assert math.isfinite(self.value), "calculation failure detected."
        self.perform_final_run()

    def post_process_iteration(self) -> None:

        v0_hat = self.current_guess
        v0 = v0_hat.no_hat()
        v0.set_name("vel_0")
        v0.set_time_step(self.i)

        write_all = os.environ.get("JAX_SPECTRAL_DNS_WRITE_FIELDS")
        out_dir = os.environ.get("JAX_SPECTRAL_DNS_FIELD_DIR")
        v0.save_to_file("velocity_latest")
        if write_all is not None and out_dir is not None:
            fname = "velocity_latest_" + str(self.i)
            copyfile(Field.field_dir + "/velocity_latest", out_dir + "/" + fname)

        # document path to the optimal
        i = self.i
        gain = self.value
        e_x_2d = v0_hat[0].energy_2d(0)
        e_z_2d = v0_hat[0].energy_2d(2)
        e_x_3d = v0.energy() - e_x_2d
        e_z_3d = v0.energy() - e_z_2d
        phase_space_data_name = Field.plotting_dir + "phase_space_data.txt"
        localisation = v0.get_localisation()
        localisation_x = (
            v0.definite_integral(2).definite_integral(1)
            / (2.0 * v0.get_physical_domain().scale_factors[2])
        ).get_localisation()
        localisation_y = (
            v0.definite_integral(2).definite_integral(0)
            / (
                v0.get_physical_domain().scale_factors[0]
                * v0.get_physical_domain().scale_factors[2]
            )
        ).get_localisation()
        localisation_z = (
            v0.definite_integral(1).definite_integral(0)
            / (2.0 * v0.get_physical_domain().scale_factors[0])
        ).get_localisation()
        with open(phase_space_data_name, "a") as file:
            file.write(
                str(i)
                + ", "
                + str(gain)
                + ", "
                + str(e_x_2d)
                + ", "
                + str(e_z_2d)
                + ", "
                + str(e_x_3d)
                + ", "
                + str(e_z_3d)
                + ", "
                + str(localisation)
                + ", "
                + str(localisation_x)
                + "\n"
            )

        # plot current state
        v0.plot_3d(0)
        v0.plot_3d(2)
        v0.plot_isosurfaces(0.6)

        phase_space_data = np.atleast_2d(
            np.genfromtxt(
                phase_space_data_name,
                delimiter=",",
            )
        ).T
        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ax.plot(
            phase_space_data[2][0], phase_space_data[3][0], "k+", label="initial guess"
        )
        ax.plot(phase_space_data[2], phase_space_data[3], "g--")
        ax.set_xlabel("$E_{2d_{x}} / E_{3d}$")
        ax.set_ylabel("$E_{2d_{z}} / E_{3d}$")
        ax.set_xlim((-0.1, 1.1))
        ax.set_yscale("log")
        fig.savefig(Field.plotting_dir + "/plot_phase_space.png")
        ax.set_ylim((1e-10, 1.1))
        ax.plot(
            phase_space_data[2][-1],
            phase_space_data[3][-1],
            "bo",
            label="current guess",
        )
        fig.savefig(
            Field.plotting_dir
            + "/plot_phase_space_t_"
            + "{:06}".format(self.i)
            + ".png"
        )

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ax.plot(phase_space_data[0], phase_space_data[1], "k.")
        ax.set_xlabel("$i$")
        ax.set_ylabel("$G$")
        fig.savefig(Field.plotting_dir + "/plot_gain_over_iterations.png")
        ax.plot(phase_space_data[0][-1], phase_space_data[1][-1], "bo")
        fig.savefig(
            Field.plotting_dir
            + "/plot_gain_over_iterations_t_"
            + "{:06}".format(self.i)
            + ".png"
        )

        fig = figure.Figure()
        ax = fig.subplots(1, 1)
        assert type(ax) is Axes
        ax.plot(phase_space_data[0], phase_space_data[2], "k.", label="$E_{x, 2d}$")
        ax.plot(phase_space_data[0], phase_space_data[4], "b.", label="$E_{3d}$")
        ax.set_xlabel("$i$")
        ax.set_ylabel("$E$")
        fig.legend()
        fig.savefig(Field.plotting_dir + "/plot_e_2d_x_over_iterations.png")
        ax.plot(phase_space_data[0][-1], phase_space_data[2][-1], "ko")
        ax.plot(phase_space_data[0][-1], phase_space_data[4][-1], "bo")
        fig.savefig(
            Field.plotting_dir
            + "/plot_e_2d_x_over_iterations_t_"
            + "{:06}".format(self.i)
            + ".png"
        )

        v0.plot_wavenumbers(1)
        v0.magnitude().plot_wavenumbers(1)

        # save state for easy restarting
        fname = Field.field_dir + "/velocity_latest"
        if (
            os.path.isfile(fname) and os.stat(fname).st_blocks > 1
        ):  # only back up velocity if it contains data
            try:
                for f in glob.glob(fname + "_bak_*"):
                    os.remove(f)
            except FileNotFoundError:
                pass
        try:
            os.rename(
                fname,
                fname + "_bak_" + str(self.i),
            )
        except FileNotFoundError:
            pass
        v0.save_to_file("velocity_latest")

    def perform_final_run(self) -> None:
        print_verb("performing final run with optimised initial condition")
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        self.dual_problem.forward_equation.set_initial_field("velocity_hat", v0_hat)
        self.dual_problem.forward_equation.write_intermediate_output = True
        self.dual_problem.forward_equation.write_entire_output = False
        self.dual_problem.forward_equation.end_time = -self.dual_problem.end_time
        self.dual_problem.forward_equation.activate_jit()
        self.dual_problem.forward_equation.set_post_process_fn(self.post_process_fn)
        self.dual_problem.forward_equation.solve()
        gain = (
            self.dual_problem.forward_equation.get_latest_field("velocity_hat")
            .no_hat()
            .energy()
            / self.dual_problem.forward_equation.get_initial_field("velocity_hat")
            .no_hat()
            .energy()
        )
        print_verb("final gain:", gain)
        self.dual_problem.forward_equation.post_process()

    def normalize_field(
        self, v0_hat: VectorField[FourierField]
    ) -> VectorField[FourierField]:
        v0 = v0_hat.no_hat()
        v0_norm = v0.normalize_by_energy()
        v0_norm *= self.e_0 ** (1 / 2)
        print_verb("energy after normalisation:", v0_norm.energy())
        v0_norm_hat = v0_norm.hat()
        v0_norm_hat.set_name("velocity_hat")
        return v0_norm_hat

    def normalize_current_guess(self) -> None:
        v0_hat = self.current_guess
        v0_norm_hat = self.normalize_field(v0_hat)
        self.current_guess = v0_norm_hat

    def determine_last_iteration_step(self) -> None:
        try:
            phase_space_data_name = Field.plotting_dir + "phase_space_data.txt"
            with open(phase_space_data_name, "r") as file:
                data = np.atleast_2d(
                    np.genfromtxt(phase_space_data_name, delimiter=",")
                ).T
                self.i = int(data[0][-1]) + 1
        except FileNotFoundError:
            print_verb(
                "file",
                phase_space_data_name,
                "not found, unable to determine last iteration step.",
            )


class SteepestAdaptiveDescentSolver(GradientDescentSolver):

    def initialise(self, prepare_for_iterations: bool = True) -> None:
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse = self.dual_problem.forward_equation
        self.dual_problem = (
            NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(nse)
        )

        if prepare_for_iterations:
            # self.value = self.dual_problem.get_gain()
            self.value = self.dual_problem.get_objective_fun()
            v_T = self.dual_problem.forward_equation.get_latest_field(
                "velocity_hat"
            ).no_hat()
            v_base = self.dual_problem.forward_equation.get_initial_field(
                "velocity_base_hat"
            ).no_hat()
            V_T = v_T + v_base

            dt = Equation.find_suitable_dt(
                self.dual_problem.forward_equation.get_physical_domain(),
                self.dual_problem.forward_equation.max_cfl,
                tuple([V_T[i].max() for i in range(3)]),
                self.dual_problem.forward_equation.end_time,
            )
            self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)
            self.old_value = self.value
            self.old_nse_dual = self.dual_problem
            self.dual_problem.forward_equation.update_dt(dt)
            self.dual_problem.update_dt(-self.dual_problem.forward_equation.get_dt())
            print_verb("")
            print_verb(self.dual_problem.get_objective_fun_name(), self.value)
            print_verb("")

    def update(self) -> None:

        v0_hat = self.current_guess
        domain = v0_hat.get_physical_domain()
        iteration_successful = False
        j = 0
        while not iteration_successful:
            start_time = time.time()
            print_verb("iteration", self.i + 1, "of at most", self.number_of_steps)
            if j + 1 > 1:
                print_verb(
                    "sub-iteration",
                    j + 1,
                    "of at most",
                    self.max_number_of_sub_iterations,
                )
            print_verb("step size:", self.step_size)

            v0_hat_new: VectorField[FourierField] = VectorField.FromData(
                FourierField,
                domain,
                v0_hat.get_data() + self.step_size * self.grad,
                name="velocity_hat",
            )
            v0_hat_new = self.normalize_field(
                v0_hat_new
            )  # should not be necessary but is done for good measure

            v0_hat_new.set_name("velocity_hat")
            nse_ = self.dual_problem.forward_equation
            Re = nse_.get_Re_tau() * nse_.get_u_max_over_u_tau()
            dt = nse_.get_dt()
            end_time = nse_.end_time
            nse = NavierStokesVelVortPerturbation(
                v0_hat_new,
                Re=Re,
                dt=dt,
                end_time=end_time,
                velocity_base_hat=nse_.get_latest_field("velocity_base_hat"),
            )
            self.dual_problem = (
                NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
                    nse
                )
            )

            gain = self.dual_problem.get_objective_fun()
            print_verb("")
            print_verb(self.dual_problem.get_objective_fun_name(), gain)

            if self.old_value is not None:
                gain_change = gain - self.old_value
                print_verb(
                    self.dual_problem.get_objective_fun_name(), "change:", gain_change
                )
            else:
                gain_change = None
            print_verb("")

            assert gain_change is not None
            if gain_change > 0.0:
                iteration_successful = True
                self.increase_step_size()
                self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)
                grad_field: VectorField[FourierField] = VectorField.FromData(
                    FourierField, domain, self.grad, name="grad_hat"
                )
                grad_nh = grad_field.no_hat()
                grad_nh.set_name("grad")
                grad_nh.plot_3d(2)
                grad_nh[0].plot_center(1)
            else:
                self.decrease_step_size()
                assert self.old_nse_dual is not None
                self.grad, _ = self.old_nse_dual.get_projected_grad(self.step_size)
                print_verb(
                    "gain decrease/stagnation detected, repeating iteration with smaller step size."
                )
            j += 1
            if j > self.max_number_of_sub_iterations:
                iteration_successful = True
            iteration_duration = time.time() - start_time
            try:
                print_verb("sub-iteration took", format_timespan(iteration_duration))
            except Exception:
                print_verb("sub-iteration took", iteration_duration, "seconds")
            print_verb("\n")

        u_T = self.dual_problem.forward_equation.get_latest_field(
            "velocity_hat"
        ).no_hat()
        u_base = self.dual_problem.forward_equation.get_initial_field(
            "velocity_base_hat"
        ).no_hat()
        u_T = u_T + u_base

        dt = Equation.find_suitable_dt(
            domain,
            self.dual_problem.forward_equation.max_cfl,
            tuple([u_T[i].max() for i in range(3)]),
            self.dual_problem.forward_equation.end_time,
        )
        self.dual_problem.forward_equation.update_dt(dt)
        self.dual_problem.update_dt(-self.dual_problem.forward_equation.get_dt())
        self.current_guess = v0_hat_new
        self.normalize_current_guess()
        v0 = self.current_guess.no_hat()
        print_verb("v0 energy:", v0.energy(), verbosity_level=2)
        self.old_nse_dual = self.dual_problem
        self.value = gain
        self.old_value = self.value


class ConjugateGradientDescentSolver(GradientDescentSolver):

    def initialise(self, prepare_for_iterations: bool = True) -> None:
        self.beta = 0.0
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        self.dual_problem.update_with_nse()

        self.value: Optional["float"] = None
        self.old_value: Optional["float"] = None
        self.old_grad: Optional["jnp_array"] = None

    def get_step_size_ls(self, old_value: "float", max_iter_ls: "int" = 10) -> "float":

        step_size = self.step_size
        print_verb("performing line search, step size", step_size)

        u_hat_0 = self.dual_problem.velocity_u_hat_0
        v_hat_0 = self.dual_problem.get_latest_field("velocity_hat")
        if self.old_grad is not None:
            self.grad, _ = self.dual_problem.get_projected_cg_grad(
                step_size, self.beta, self.old_grad, u_hat_0, v_hat_0
            )
        else:
            self.grad, _ = self.dual_problem.get_projected_grad(
                step_size, u_hat_0, v_hat_0
            )
        current_guess = self.current_guess + step_size * self.grad
        current_guess = self.normalize_field(current_guess)

        self.dual_problem.forward_equation.set_initial_field(
            "velocity_hat", current_guess
        )
        self.dual_problem.update_with_nse()
        self.dual_problem.write_trajectory = False
        new_value = self.dual_problem.get_objective_fun()
        self.dual_problem.write_trajectory = True
        print_verb("gain:", new_value)

        local_grad = self.dual_problem.get_projected_grad(step_size, u_hat_0, v_hat_0)[
            0
        ].flatten()
        tau = 0.5
        c = 0.5
        j = 1
        # m = jax.numpy.linalg.norm(self.grad)
        # m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (self.e_0 / old_value)
        m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (
            self.e_0 / old_value**2
        )
        t = c * m
        cond = new_value - old_value > step_size * t
        print_verb("m", m)
        print_verb("t", t)
        print_verb("step_size * t", step_size * t)
        if cond:
            print_verb("wolfe conditions satisfied, trying to increase the step size")
            while new_value - old_value > step_size * t and j < max_iter_ls:
                step_size /= tau
                print_verb("line search iteration", j, "step size", step_size)

                if self.old_grad is not None:
                    self.grad, _ = self.dual_problem.get_projected_cg_grad(
                        step_size, self.beta, self.old_grad, u_hat_0, v_hat_0
                    )
                else:
                    self.grad, _ = self.dual_problem.get_projected_grad(
                        step_size, u_hat_0, v_hat_0
                    )
                current_guess = self.current_guess + step_size * self.grad
                current_guess = self.normalize_field(current_guess)

                # TODO: possibly recompute gradient
                self.dual_problem.forward_equation.set_initial_field(
                    "velocity_hat", current_guess
                )
                self.dual_problem.update_with_nse()
                self.dual_problem.write_trajectory = False
                new_value = self.dual_problem.get_objective_fun()
                self.dual_problem.write_trajectory = True
                print_verb("gain:", new_value)
                # m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (
                #     self.e_0 / old_value
                # )
                # m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (1.0 / old_value) ** 2
                m = jnp.abs(
                    jnp.dot(
                        self.dual_problem.get_projected_grad(
                            step_size, u_hat_0, v_hat_0
                        )[0].flatten(),
                        self.grad.flatten(),
                    )
                ) * (self.e_0 / old_value**2)
                t = c * m
                j += 1
                print_verb("m", m)
                print_verb("t", t)
                print_verb("step_size * t", step_size * t)
            step_size *= tau

        else:
            print_verb(
                "wolfe conditions not satisfied, trying to decrease the step size"
            )
            while new_value - old_value < step_size * t and j < max_iter_ls:
                step_size *= tau

                print_verb("line search iteration", j, "step size", step_size)

                if self.old_grad is not None:
                    self.grad, _ = self.dual_problem.get_projected_cg_grad(
                        step_size, self.beta, self.old_grad, u_hat_0, v_hat_0
                    )
                else:
                    self.grad, _ = self.dual_problem.get_projected_grad(
                        step_size, u_hat_0, v_hat_0
                    )
                current_guess = self.current_guess + step_size * self.grad
                current_guess = self.normalize_field(current_guess)

                # TODO: possibly recompute gradient
                self.dual_problem.forward_equation.set_initial_field(
                    "velocity_hat", current_guess
                )
                self.dual_problem.update_with_nse()
                self.dual_problem.write_trajectory = False
                new_value = self.dual_problem.get_objective_fun()
                self.dual_problem.write_trajectory = True
                print_verb("gain:", new_value)
                # m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (
                #     self.e_0 / old_value
                # )
                # m = jnp.abs(jnp.dot(local_grad, self.grad.flatten())) * (1.0 / old_value) ** 2
                m = jnp.abs(
                    jnp.dot(
                        self.dual_problem.get_projected_grad(
                            step_size, u_hat_0, v_hat_0
                        )[0].flatten(),
                        self.grad.flatten(),
                    )
                ) * (self.e_0 / old_value**2)
                t = c * m
                j += 1
                print_verb("m", m)
                print_verb("t", t)
                print_verb("step_size * t", step_size * t)
        return cast(float, step_size)

    def update(self) -> None:

        start_time = time.time()
        print_verb("iteration", self.i + 1, "of", self.number_of_steps)
        print_verb("step size:", self.step_size, "; beta:", self.beta)

        v0 = self.current_guess.no_hat()
        v0_div = v0.div()
        cont_error = v0_div.energy() / v0.energy()
        print_verb("cont_error", cont_error)

        if self.i % self.trajectory_write_interval == 0:
            self.dual_problem.write_trajectory = True
        else:
            self.dual_problem.write_trajectory = False

        domain = self.dual_problem.get_physical_domain()

        self.dual_problem.forward_equation.set_initial_field(
            "velocity_hat", self.current_guess
        )
        self.dual_problem.update_with_nse()
        gain = self.dual_problem.get_objective_fun()

        print_verb("")
        print_verb(self.dual_problem.get_objective_fun_name(), gain)
        if self.value is not None:
            gain_change = gain - self.value
            print_verb(
                self.dual_problem.get_objective_fun_name(), "change:", gain_change
            )
        else:
            gain_change = None
        print_verb("")

        if self.old_grad is not None:
            self.grad, _ = self.dual_problem.get_projected_cg_grad(
                self.step_size, self.beta, self.old_grad
            )
        else:
            self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)

        if self.use_linesearch:
            self.step_size = self.get_step_size_ls(gain)

        self.current_guess = self.current_guess + self.step_size * self.grad
        self.current_guess = self.normalize_field(self.current_guess)
        if self.dual_problem.forward_equation.constant_mass_flux:
            self.current_guess = (
                self.dual_problem.forward_equation.update_velocity_field(
                    self.current_guess
                )
            )

        if Equation.verbosity_level >= 3:
            grad_field: VectorField[FourierField] = VectorField.FromData(
                FourierField, domain, self.grad, name="grad_hat"
            )
            print_verb("grad energy:", grad_field.no_hat().energy(), verbosity_level=3)
            print_verb(
                "grad energy times step size:",
                (self.step_size * grad_field).no_hat().energy(),
                verbosity_level=3,
            )

        if gain_change is not None:
            if gain_change > 0.0:
                if not self.use_linesearch:
                    self.increase_step_size()
                self.reset_beta = False
            else:
                if not self.use_linesearch:
                    self.decrease_step_size()
                self.reset_beta = True
            self.update_beta(not self.reset_beta)
        self.old_grad = self.grad
        self.old_value = self.value
        self.value = gain

        print_verb(
            "current flow rate:",
            self.dual_problem.forward_equation.get_flow_rate(
                self.current_guess.get_data()
            ),
        )

        iteration_duration = time.time() - start_time
        try:
            print_verb("sub-iteration took", format_timespan(iteration_duration))
        except Exception:
            print_verb("sub-iteration took", iteration_duration, "seconds")
        print_verb("\n")

    def decrease_step_size(self, tau: float = 2.0) -> None:
        self.step_size = max(self.step_size / tau, self.min_step_size)

    def update_beta(self, last_iteration_successful: bool) -> None:
        if last_iteration_successful:
            assert self.old_grad is not None
            grad = self.grad.flatten()
            old_grad = self.old_grad.flatten()
            self.beta = cast(
                float, jnp.dot(grad, (grad - old_grad)) / jnp.dot(old_grad, old_grad)
            )
            if self.beta.real < 0.0:
                self.beta = 0.0
        else:
            self.beta = 0.0


class OptimiserWrapper(GradientDescentSolver):

    def initialise(self, _: bool = True) -> None:
        print_verb("using optimiser from external library")

        def run_input_to_parameters(x: VectorField[FourierField]) -> "parameter_type":
            return (x.get_data(),)

        def parameters_to_run_input(x: "parameter_type") -> VectorField[FourierField]:
            vel_0_hat: VectorField[FourierField] = VectorField.FromData(
                FourierField,
                self.dual_problem.get_physical_domain(),
                x[0],
                "velocity_hat",
            )
            vel_0_hat.set_name("velocity_hat")
            return vel_0_hat

        def run_adjoint(
            v0_hat: VectorField[FourierField], out: bool = False
        ) -> Tuple[float, Tuple["jnp_array"]]:
            v0 = v0_hat.no_hat()
            v0 = v0.normalize_by_energy()
            v0 *= self.e_0
            v0_hat_ = v0.hat()
            v0_hat_.set_name("velocity_hat")
            Re_tau = self.dual_problem.forward_equation.get_Re_tau()
            dt = self.dual_problem.forward_equation.get_dt()
            end_time = self.dual_problem.forward_equation.end_time
            nse = NavierStokesVelVortPerturbation(
                v0_hat_, Re_tau=Re_tau, dt=dt, end_time=end_time
            )
            nse_dual = (
                NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
                    nse
                )
            )
            if out:
                return (nse_dual.get_objective_fun(), (jnp.array([0.0]),))
            else:
                return (nse_dual.get_objective_fun(), (-1 * nse_dual.get_grad(),))

        run_input_initial = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        )

        self.optimiser = OptimiserFourier(
            self.dual_problem.get_physical_domain(),
            self.dual_problem.get_physical_domain(),
            run_adjoint,
            run_input_initial,
            value_and_grad=True,
            minimise=False,
            force_2d=False,
            max_iter=self.number_of_steps,
            use_optax=False,
            min_optax_iter=self.number_of_steps,
            learning_rate=1e-2,
            scale_by_norm=True,
            objective_fn_name="gain",
            add_noise=False,
            parameters_to_run_input_fn=parameters_to_run_input,
            run_input_to_parameters_fn=run_input_to_parameters,
        )

    def update(self) -> None:
        self.optimiser.current_iteration = self.i
        self.optimiser.perform_iteration()
        self.current_guess = self.optimiser.parameters_to_run_input_(
            self.optimiser.parameters
        )
        self.value = cast("float", self.optimiser.value)
        self.normalize_current_guess()

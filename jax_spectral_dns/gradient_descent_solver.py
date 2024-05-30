#!/usr/bin/env python

from abc import ABC, abstractmethod
import time
import math
from typing import Any, cast
import jax
import jax.numpy as jnp
from matplotlib.axes import subplot_class_factory
from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import FourierField, VectorField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation
from jax_spectral_dns.navier_stokes_perturbation_dual import (
    NavierStokesVelVortPerturbationDual,
)

try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    pass


class GradientDescentSolver(ABC):

    def __init__(
        self, dual_problem: NavierStokesVelVortPerturbationDual, **params: Any
    ):
        self.dual_problem = dual_problem
        self.max_step_size = params.get("max_step_size", 0.999)
        self.step_size = params.get("step_size", 1e-2)
        self.number_of_steps = params.get("max_iterations", 20)
        self.relative_gain_increase_threshold = params.get(
            "relative_gain_increase_threshold", 0.2
        )
        self.max_number_of_sub_iterations = params.get(
            "max_number_of_sub_iterations", 100
        )
        self.current_guess = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        )

        self.i = 0
        self.e_0 = 0.0
        self.done = False
        self.post_process_fn = params.get("post_process_function", None)
        self.value = -1.0

    def increase_step_size(self) -> None:
        self.step_size = min(self.max_step_size, self.step_size * 1.5)

    def decrease_step_size(self) -> None:
        self.step_size /= 5.0

    @abstractmethod
    def initialise(self, prepare_for_iterations: bool = True) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    def optimise(self) -> None:
        self.initialise(self.number_of_steps >= 0)
        if self.number_of_steps >= 0:
            assert math.isfinite(self.value), "calculation failure detected."
        i = 0
        if i >= self.number_of_steps or self.step_size < 1e-4:
            self.done = True
        while not self.done:
            self.i = i
            self.update()
            self.post_process_iteration()
            i += 1
            if i >= self.number_of_steps or self.step_size < 1e-20:
                self.done = True
            assert math.isfinite(self.value), "calculation failure detected."
        self.perform_final_run()

    def post_process_iteration(self) -> None:
        v0_hat = self.current_guess
        v0 = v0_hat.no_hat()
        v0.set_name("vel_0")
        v0.set_time_step(self.i)
        v0.plot_3d(2)
        v0[0].plot_center(1)
        v0.save_to_file("velocity_latest")

    def perform_final_run(self) -> None:
        print_verb("performing final run with optimised initial condition")
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse_ = self.dual_problem.forward_equation
        Re = nse_.get_Re_tau() * nse_.get_u_max_over_u_tau()
        dt = nse_.get_dt()
        end_time = nse_.end_time

        nse = NavierStokesVelVortPerturbation(
            v0_hat,
            Re=Re,
            dt=dt,
            end_time=end_time,
            velocity_base_hat=nse_.get_latest_field("velocity_base_hat"),
        )
        nse.set_linearize(False)
        nse.write_intermediate_output = True
        nse.activate_jit()
        nse.set_post_process_fn(self.post_process_fn)
        nse.solve()
        nse.post_process()

    def normalize_field(
        self, v0_hat: VectorField[FourierField]
    ) -> VectorField[FourierField]:
        v0 = v0_hat.no_hat()
        v0_norm = v0.normalize_by_energy()
        v0_norm *= self.e_0 ** (1 / 2)
        v0_norm_hat = v0_norm.hat()
        v0_norm_hat.set_name("velocity_hat")
        return v0_norm_hat

    def normalize_current_guess(self) -> None:
        v0_hat = self.current_guess
        v0_norm_hat = self.normalize_field(v0_hat)
        self.current_guess = v0_norm_hat


class SteepestAdaptiveDescentSolver(GradientDescentSolver):

    def initialise(self, prepare_for_iterations: bool = True) -> None:
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse = self.dual_problem.forward_equation
        # nse.set_linearize(True)
        nse.set_linearize(False)
        self.dual_problem = (
            NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(nse)
        )

        self.e_0 = nse.get_initial_field("velocity_hat").no_hat().energy()

        if prepare_for_iterations:
            self.value = self.dual_problem.get_gain()
            self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)
            self.old_value = self.value
            self.old_nse_dual = self.dual_problem
            print_verb("")
            print_verb("gain:", self.value)
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
            nse.set_linearize(False)
            self.dual_problem = (
                NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
                    nse
                )
            )

            gain = self.dual_problem.get_gain()
            gain_change = gain - self.old_value
            print_verb("")
            print_verb("gain:", gain)
            print_verb("gain change:", gain_change)
            print_verb("")

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

        nse = self.dual_problem.forward_equation
        nse.set_linearize(False)
        self.dual_problem.update_with_nse()

        self.e_0 = nse.get_initial_field("velocity_hat").no_hat().energy()

        if prepare_for_iterations:
            self.value = self.dual_problem.get_gain()
            self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)
            self.old_value = self.value
            self.old_grad = self.grad
            self.old_nse_dual = self.dual_problem
            print_verb("")
            print_verb("gain:", self.value)
            print_verb("")

    def update(self) -> None:

        v0_hat = self.current_guess
        domain = v0_hat.get_physical_domain()
        iteration_successful = False
        j = 0
        success = False
        while not iteration_successful:
            start_time = time.time()
            print_verb("iteration", self.i + 1, "of", self.number_of_steps)
            if j + 1 > 1:
                print_verb("sub-iteration", j + 1)
            print_verb("step size:", self.step_size)
            print_verb("beta:", self.beta, verbosity_level=2)

            v0_hat_new: VectorField[FourierField] = VectorField.FromData(
                FourierField,
                domain,
                v0_hat.get_data() + self.step_size * self.grad,
                name="velocity_hat",
            )
            v0_hat_new = self.normalize_field(v0_hat_new)

            v0_hat_new.set_name("velocity_hat")
            self.dual_problem.forward_equation.set_initial_field(
                "velocity_hat", v0_hat_new
            )
            self.dual_problem.update_with_nse()

            gain = self.dual_problem.get_gain()
            gain_change = gain - self.old_value
            print_verb("")
            print_verb("gain:", gain)
            print_verb("gain change:", gain_change)
            print_verb("")

            gain_change_ok: bool = (gain_change > 0.0) and (
                gain_change / self.old_value < self.relative_gain_increase_threshold
            )
            if gain_change_ok:
                iteration_successful = True
                self.increase_step_size()

                self.grad, success = self.dual_problem.get_projected_cg_grad(
                    self.step_size, self.beta, self.old_grad
                )
                if not success and abs(self.beta) > 1e2:
                    print_verb(
                        "problems with finding lambda due to high beta detected, repeating gradient calculation with beta=0."
                    )
                    self.beta = 0.0
                    self.grad, success = self.dual_problem.get_projected_cg_grad(
                        self.step_size, self.beta, self.old_grad
                    )
                grad_field: VectorField[FourierField] = VectorField.FromData(
                    FourierField, domain, self.grad, name="grad_hat"
                )
                grad_nh = grad_field.no_hat()
                grad_nh.set_name("grad")
                grad_nh.plot_3d(2)
                grad_nh[0].plot_center(1)
            else:
                if gain_change <= 0.0:
                    print_verb(
                        "gain decrease/stagnation detected, repeating iteration with smaller step size."
                    )
                if (
                    gain_change / self.old_value
                    >= self.relative_gain_increase_threshold
                ):
                    print_verb(
                        "high gain increase detected, repeating iteration with smaller step size."
                    )
                self.decrease_step_size()
                assert self.old_nse_dual is not None
                if gain_change <= 0.0:
                    self.beta = 0.0
                self.grad, _ = self.old_nse_dual.get_projected_grad(self.step_size)

            j += 1
            if j > self.max_number_of_sub_iterations:
                iteration_successful = True
            iteration_duration = time.time() - start_time
            try:
                print_verb("sub-iteration took", format_timespan(iteration_duration))
            except Exception:
                print_verb("sub-iteration took", iteration_duration, "seconds")
            print_verb("\n")
            # jax.clear_caches()  # type: ignore[no-untyped-call] # TODO does this help? how much does it hurt?

        self.update_beta(True)
        self.current_guess = v0_hat_new
        self.normalize_current_guess()
        v0 = self.current_guess.no_hat()
        print_verb("v0 energy:", v0.energy(), verbosity_level=2)
        self.old_nse_dual = self.dual_problem
        self.value = gain
        self.old_value = self.value
        self.old_grad = self.grad

    def decrease_step_size(self) -> None:
        self.step_size /= 2.0

    def update_beta(self, last_iteration_successful: bool) -> None:
        if last_iteration_successful:
            grad = self.grad.flatten()
            old_grad = self.old_grad.flatten()
            self.beta = cast(
                float, jnp.dot(grad, (grad - old_grad)) / jnp.dot(old_grad, old_grad)
            )
            if self.beta.real < 0.0:
                self.beta = 0.0
        else:
            self.beta = 0.0

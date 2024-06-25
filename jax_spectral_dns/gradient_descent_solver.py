#!/usr/bin/env python

from abc import ABC, abstractmethod
import os, glob
import time
import math
from typing import Any, Tuple, cast, TYPE_CHECKING
import jax
import jax.numpy as jnp
from matplotlib.axes import subplot_class_factory
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
        self.value = -1.0
        self.old_value = self.value

        # set various solver options
        self.i = params.get("start_iteration", 0)
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
        self.value_change_threshold = params.get("value_change_threshold", 1e-7)
        self.step_size_threshold = params.get("step_size_threshold", 1e-5)

        self.current_guess = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        )

    @property
    def e_0(self) -> float:
        return self._e_0

    def increase_step_size(self) -> None:
        self.step_size = min(self.max_step_size, self.step_size * 1.5)

    def decrease_step_size(self) -> None:
        self.step_size /= 5.0

    @abstractmethod
    def initialise(self, prepare_for_iterations: bool = True) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    def is_done(self, i: int) -> bool:
        done: bool = (i >= self.number_of_steps) or (
            self.step_size < self.step_size_threshold
        )
        return done

    def update_done(self, i: int) -> None:
        if self.is_done(i):
            self.done = True

    def optimise(self) -> None:
        self.initialise(self.number_of_steps >= 0)
        if self.number_of_steps >= 0:
            assert math.isfinite(self.value), "calculation failure detected."
        self.update_done(self.i)
        while not self.done:
            self.update()
            self.post_process_iteration()
            self.i += 1
            self.update_done(self.i)
            assert math.isfinite(self.value), "calculation failure detected."
        self.perform_final_run()

    def post_process_iteration(self) -> None:
        v0_hat = self.current_guess
        v0 = v0_hat.no_hat()
        v0.set_name("vel_0")
        v0.set_time_step(self.i)
        v0.plot_3d(0)
        v0.plot_3d(2)
        v0[1].plot_isosurfaces(0.4)
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


class SteepestAdaptiveDescentSolver(GradientDescentSolver):

    def initialise(self, prepare_for_iterations: bool = True) -> None:
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse = self.dual_problem.forward_equation
        self.dual_problem = (
            NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(nse)
        )

        if prepare_for_iterations:
            self.value = self.dual_problem.get_gain()
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

        if prepare_for_iterations:
            self.almost_done = False
            self.value = self.dual_problem.get_gain()
            self.grad, _ = self.dual_problem.get_projected_grad(self.step_size)
            self.old_value = self.value
            self.old_grad = self.grad
            self.old_nse_dual = self.dual_problem
            self.v_0_hat_old = self.dual_problem.get_latest_field("velocity_hat")
            self.u_0_hat_old = v0_hat
            print_verb("")
            print_verb("gain:", self.value)
            print_verb("")

    def update(self) -> None:

        v0_hat = self.current_guess
        domain = v0_hat.get_physical_domain()
        iteration_successful = False
        break_iteration = False
        j = 0
        success = False
        while (not iteration_successful) and (not break_iteration):
            start_time = time.time()
            print_verb("iteration", self.i + 1, "of", self.number_of_steps)
            if j + 1 > 1:
                print_verb("sub-iteration", j + 1)
            print_verb("step size:", self.step_size, "; beta:", self.beta)

            v0_hat_new: VectorField[FourierField] = VectorField.FromData(
                FourierField,
                domain,
                v0_hat.get_data() + self.step_size * self.grad,
                name="velocity_hat",
            )
            v0_hat_new = self.normalize_field(v0_hat_new)

            # TODO remove this eventually
            grad_field: VectorField[FourierField] = VectorField.FromData(
                FourierField, domain, self.grad, name="grad_hat"
            )
            print_verb(
                "grad energy:",
                grad_field.no_hat().energy(),
            )
            print_verb(
                "grad energy times step size:",
                self.step_size * grad_field.no_hat().energy(),
            )

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

            gain_change_ok: bool = (
                math.isfinite(gain)
                and (gain_change > 0.0)
                and (
                    gain_change / self.old_value < self.relative_gain_increase_threshold
                )
            )
            if gain_change_ok:
                iteration_successful = True
                self.almost_done = False
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
                # if gain_change <= 1e-4:
                #     self.beta = 0.0
            else:
                # if gain_change <= 0.0:
                #     print_verb(
                #         "gain decrease/stagnation detected, repeating iteration with smaller step size."
                #     )
                # if (
                #     gain_change / self.old_value
                #     >= self.relative_gain_increase_threshold
                # ):
                #     print_verb(
                #         "high gain increase detected, repeating iteration with smaller step size."
                #     )
                # TODO is always accepting new guess a good idea?
                iteration_successful = True
                self.almost_done = False

                self.decrease_step_size()
                if gain_change <= 0.0:
                    self.beta = 0.0
                self.grad, success = self.dual_problem.get_projected_cg_grad(
                    self.step_size, self.beta, self.old_grad
                )
                # self.grad, _ = self.dual_problem.get_projected_grad_from_u_and_v(
                #     self.step_size,
                #     self.u_0_hat_old,
                #     self.v_0_hat_old,
                # )

            # make sure we stop at some point
            if (
                abs(self.step_size - self.min_step_size) <= 1e-50
                and abs(self.beta) <= 1e-50
            ):
                if self.almost_done:
                    self.done = True
                self.almost_done = True

            j += 1
            if j > self.max_number_of_sub_iterations or self.done:
                break_iteration = True
            iteration_duration = time.time() - start_time
            try:
                print_verb("sub-iteration took", format_timespan(iteration_duration))
            except Exception:
                print_verb("sub-iteration took", iteration_duration, "seconds")
            print_verb("\n")

        if iteration_successful:
            self.update_beta(True)
            self.current_guess = v0_hat_new
            # self.normalize_current_guess()
            self.value = gain

            self.old_value = self.value
            self.old_grad = self.grad
            self.v_0_hat_old = self.dual_problem.get_latest_field("velocity_hat")
            self.u_0_hat_old = v0_hat_new

    def decrease_step_size(self) -> None:
        self.step_size = max(self.step_size / 2.0, self.min_step_size)

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
            nse.set_linearize(False)
            nse_dual = (
                NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
                    nse
                )
            )
            if out:
                return (nse_dual.get_gain(), (jnp.array([0.0]),))
            else:
                return (nse_dual.get_gain(), (-1 * nse_dual.get_grad(),))

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
            use_optax=True,
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

#!/usr/bin/env python

from abc import ABC, abstractmethod
import time
from typing import Any
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
        self.number_of_steps = params.get("number_of_steps", 10)
        self.current_guess = self.dual_problem.forward_equation.get_initial_field(
            "velocity_hat"
        )
        self.i = 0
        self.number_of_steps = params.get("max_iterations", 20)

    def increase_step_size(self) -> None:
        self.step_size = min(self.max_step_size, self.step_size * 1.5)

    def decrease_step_size(self) -> None:
        self.step_size /= 5.0

    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    def optimise(self) -> None:
        self.init()
        for i in range(self.number_of_steps):
            self.i = i
            self.update()
            self.post_process_iteration()
        self.perform_final_run()

    def post_process_iteration(self) -> None:
        v0_hat = self.current_guess
        v0 = v0_hat.no_hat()
        v0.set_name("vel_0")
        v0.set_time_step(self.i)
        v0.plot_3d(2)
        v0[0].plot_center(1)

    def perform_final_run(self) -> None:
        print_verb("performing final run with optimised initial condition.")
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse_ = self.dual_problem.forward_equation
        Re = nse_.get_Re_tau() * nse_.get_u_max_over_u_tau()
        dt = nse_.get_dt()
        end_time = nse_.end_time

        nse = NavierStokesVelVortPerturbation(v0_hat, Re=Re, dt=dt, end_time=end_time)
        nse.set_linearize(False)
        nse.write_intermediate_output = True
        nse.activate_jit()
        nse.solve()
        nse.post_process()


class SteepestAdaptiveDescentSolver(GradientDescentSolver):

    def initialise(self) -> None:
        v0_hat = self.current_guess
        v0_hat.set_name("velocity_hat")

        nse = self.dual_problem.forward_equation
        # nse.set_linearize(True)
        nse.set_linearize(False)
        nse_dual = (
            NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(nse)
        )

        self.e_0 = nse.get_initial_field("velocity_hat").no_hat().energy()

        self.value = nse_dual.get_gain()
        self.grad = nse_dual.get_projected_grad(self.step_size)
        self.old_value = self.value
        self.old_nse_dual = nse_dual

    def update(self) -> None:

        v0_hat = self.current_guess
        domain = v0_hat.get_physical_domain()
        iteration_successful = False
        j = 0
        while not iteration_successful:
            start_time = time.time()
            print_verb("iteration", self.i + 1, "of", self.number_of_steps)
            print_verb("sub-iteration", j + 1)
            print_verb("step size:", self.step_size)

            v0_hat_new: VectorField[FourierField] = VectorField.FromData(
                FourierField,
                domain,
                v0_hat.get_data() + self.step_size * self.grad,
                name="velocity_hat",
            )

            v0_hat_new.set_name("velocity_hat")
            nse_ = self.dual_problem.forward_equation
            Re = nse_.get_Re_tau() * nse_.get_u_max_over_u_tau()
            dt = nse_.get_dt()
            end_time = nse_.end_time
            nse = NavierStokesVelVortPerturbation(
                v0_hat_new, Re=Re, dt=dt, end_time=end_time
            )
            # nse.set_linearize(True)
            nse.set_linearize(False)
            nse_dual = (
                NavierStokesVelVortPerturbationDual.FromNavierStokesVelVortPerturbation(
                    nse
                )
            )

            gain = nse_dual.get_gain()
            gain_change = gain - self.old_value
            print_verb("")
            print_verb("gain:", gain)
            print_verb("gain change:", gain_change)
            print_verb("")

            if gain_change > 0.0:
                iteration_successful = True
                self.increase_step_size()
                self.grad = nse_dual.get_projected_grad(self.step_size)
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
                self.grad = self.old_nse_dual.get_projected_grad(self.step_size)
                print_verb(
                    "gain decrease/stagnation detected, repeating iteration with smaller step size."
                )

            j += 1
            iteration_duration = time.time() - start_time
            try:
                print_verb("sub-iteration took", format_timespan(iteration_duration))
            except Exception:
                print_verb("sub-iteration took", iteration_duration, "seconds")
            print_verb("\n")

        self.current_guess = v0_hat_new
        v0 = self.current_guess.no_hat()
        energy = v0.energy()
        print_verb("v0 energy:", energy)
        v0 = v0.normalize_by_energy()
        v0 *= self.e_0**0.5
        v0_hat = v0.hat()
        v0_hat.set_name("velocity_hat")
        self.old_nse_dual = nse_dual
        self.old_value = self.value
        self.value = gain

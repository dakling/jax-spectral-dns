#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Union, TYPE_CHECKING, cast
import jax
import jax.numpy as jnp
import pickle

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import Field, FourierField, PhysicalField, VectorField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

from jax_spectral_dns._typing import jsd_float, parameter_type, jnp_array

if TYPE_CHECKING:
    from jax_spectral_dns._typing import AnyVectorField, input_type

try:
    import optax  # type: ignore
except ModuleNotFoundError:
    print("optax not found")
try:
    import jaxopt  # type: ignore
except ModuleNotFoundError:
    print("jaxopt not found")


class Optimiser:

    def __init__(
        self,
        domain: PhysicalDomain,
        run_fn: Callable[["input_type", bool], jsd_float],
        run_input_initial: Union["input_type", str],
        minimise: bool = False,
        force_2d: bool = False,
        max_iter: int = 20,
        use_optax: bool = False,
        min_optax_iter: int = 0,
        add_noise: bool = True,
        noise_amplitude: float = 1e-1,
        **params: Any,
    ):

        self.parameters_to_run_input_fn = params.get("parameters_to_run_input_fn")
        self.run_input_to_parameters_fn = params.get("run_input_to_parameters_fn")

        self.domain = domain
        self.force_2d = force_2d
        if type(run_input_initial) is str:
            # we would like to be flexible with how the input is provided - see comments for details
            if Field.field_dir in run_input_initial:  # e.g. "./fields/parameters"
                self.parameter_file_name = run_input_initial.split(Field.field_dir)[-1]
            elif Field.field_dir[2:] in run_input_initial:  # e.g. "fields/parameters"
                assert (
                    Field.field_dir[1:] not in run_input_initial
                ), "Directory not found."  # checks that it is not e.g. "/fields/parameters"
                self.parameter_file_name = run_input_initial.split(Field.field_dir)[-1]
            else:  # e.g. just "parameters"
                self.parameter_file_name = run_input_initial
            self.parameters = self.parameters_from_file()
        else:
            self.parameter_file_name = "parameters"
            if add_noise:
                run_input: "input_type" = self.make_noisy(
                    cast(input_type, run_input_initial), noise_amplitude=noise_amplitude
                )
            else:
                run_input = cast(input_type, run_input_initial)
            self.parameters = self.run_input_to_parameters(run_input)
        self.old_value: Optional[jsd_float] = None
        self.current_iteration: int = 0
        self.max_iter: int = max_iter
        self.min_optax_iter: int = min_optax_iter
        if minimise:
            self.inv_fn: Callable[[jsd_float], jsd_float] = lambda x: x
        else:
            self.inv_fn = lambda x: -x
        self.run_fn: Callable[[parameter_type, bool], jsd_float] = lambda v, out=False: self.inv_fn(  # type: ignore[misc]
            run_fn(self.parameters_to_run_input(v), out)
        )
        if use_optax:
            learning_rate = params.get("learning_rate", 1e-2)
            scale_by_norm = params.get("scale_by_norm", True)
            self.solver = self.get_optax_solver(learning_rate, scale_by_norm)
            self.solver_switched = False
            print_verb("Using Optax solver")
        else:
            self.solver = self.get_jaxopt_solver()
            self.solver_switched = True
            print_verb("Using jaxopt solver")
        self.state = self.solver.init_state(self.parameters)
        self.objective_fn_name = params.get("objective_fn_name", "objective function")

        self.value = self.inv_fn(self.state.value)
        print_verb(self.objective_fn_name + ":", self.value)

    def parameters_to_run_input(self, parameters: parameter_type) -> input_type:
        if self.parameters_to_run_input_fn == None:
            if self.force_2d:
                vort_hat: Optional[jnp_array] = None
                v1_hat: jnp_array = parameters[0]
                v0_00: jnp_array = parameters[1]
                v2_00: Optional[jnp_array] = None
            else:
                vort_hat = parameters[0]
                v1_hat = parameters[1]
                v0_00 = parameters[2]
                v2_00 = parameters[3]
            domain = self.domain
            U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
                domain, vort_hat, v1_hat, v0_00, v2_00, two_d=self.force_2d
            )
            input: VectorField[FourierField] = VectorField.FromData(
                FourierField, domain, U_hat_data
            )
        else:
            assert self.parameters_to_run_input_fn is not None
            input = self.parameters_to_run_input_fn(parameters)

        def set_time_step_rec(inp: Any) -> None:
            if isinstance(inp, Field) or isinstance(inp, VectorField):
                inp.set_time_step(self.current_iteration)
            else:
                for inp_i in inp:
                    set_time_step_rec(inp_i)

        set_time_step_rec(input)
        return input

    def parameters_from_file(self) -> parameter_type:
        """Load paramters from file filename."""
        print_verb("loading parameters from", self.parameter_file_name)
        with open(Field.field_dir + self.parameter_file_name, "rb") as file:
            self.parameters = pickle.load(file)
        return self.parameters

    def parameters_to_file(self) -> None:
        """Save paramters to file filename."""
        with open(Field.field_dir + self.parameter_file_name, "wb") as file:
            pickle.dump(self.parameters, file, protocol=pickle.HIGHEST_PROTOCOL)

    def run_input_to_parameters(self, input: "input_type") -> parameter_type:
        input_ = cast(VectorField[FourierField], input)
        if self.run_input_to_parameters_fn == None:
            if self.force_2d:
                v0_1 = input_[1].data * (1 + 0j)
                v0_0_00_hat = input_[0].data[0, :, 0] * (1 + 0j)
                self.parameters = tuple([v0_1, v0_0_00_hat])
            else:
                vort_hat = input_.curl()[1].data * (1 + 0j)
                v0_1 = input_[1].data * (1 + 0j)
                v0_0_00_hat = input_[0].data[0, :, 0] * (1 + 0j)
                v2_0_00_hat = input_[2].data[0, :, 0] * (1 + 0j)
                self.parameters = (vort_hat, v0_1, v0_0_00_hat, v2_0_00_hat)
        else:
            assert self.run_input_to_parameters_fn is not None
            self.parameters = self.run_input_to_parameters_fn(input_)
        return self.parameters

    def get_parameters_norm(self) -> jsd_float:
        return jnp.linalg.norm(
            jnp.concatenate([jnp.array(v.flatten()) for v in self.parameters])
        )

    def make_noisy(
        self, input: "input_type", noise_amplitude: jsd_float = 1e-1
    ) -> "input_type":
        input_ = cast(VectorField[FourierField], input)
        parameters_no_hat = input_.no_hat()
        e0 = parameters_no_hat.energy()
        interval_bound = e0**0.5 * noise_amplitude / 2
        return VectorField(
            [
                f
                + FourierField.FromRandom(
                    self.domain, seed=37, interval=(-interval_bound, interval_bound)
                )
                for f in input_
            ]
        )

    def get_optax_solver(
        self, learning_rate: jsd_float = 1e-2, scale_by_norm: bool = True
    ) -> jaxopt.OptaxSolver:
        learning_rate_ = (
            learning_rate * self.get_parameters_norm()
            if scale_by_norm
            else learning_rate
        )
        opt = optax.adam(learning_rate=learning_rate_)  # minimizer
        solver = jaxopt.OptaxSolver(
            opt=opt, fun=jax.value_and_grad(self.run_fn), value_and_grad=True, jit=True
        )
        return solver

    def get_jaxopt_solver(self) -> jaxopt.LBFGS:
        solver = jaxopt.LBFGS(
            jax.value_and_grad(self.run_fn),
            value_and_grad=True,
            implicit_diff=True,
            jit=True,
            linesearch="zoom",
            linesearch_init="current",
            maxls=15,
        )
        return solver

    def post_process_iteration(self) -> None:

        i = self.current_iteration
        U_hat = cast(
            VectorField[FourierField], self.parameters_to_run_input(self.parameters)
        )
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        cont_error = v0_new.div().energy() / v0_new.energy()
        print_verb("cont_error", cont_error)

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(0)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        v0_new[0].plot_isosurfaces(0.4)
        self.parameters_to_file()

    def switch_solver_maybe(self) -> None:
        i = self.current_iteration
        min_number_of_optax_steps = self.min_optax_iter
        assert self.old_value is not None
        if (
            (not self.solver_switched)
            and i >= min_number_of_optax_steps
            and self.value - self.old_value < 0
        ):
            print_verb("switching to jaxopt solver")
            self.solver = self.get_jaxopt_solver()
            self.solver_switched = True
            self.state = self.solver.init_state(self.parameters)

    def perform_iteration(self) -> None:
        start_time = time.time()
        i = self.current_iteration
        number_of_steps = self.max_iter
        print_verb("Iteration", i + 1, "of", number_of_steps)

        solver = self.solver

        self.post_process_iteration()

        self.parameters, self.state = solver.update(self.parameters, self.state)
        inverse_value = self.state.value
        new_value = self.inv_fn(inverse_value)
        self.old_value = self.value
        self.value = new_value
        print_verb()
        print_verb(self.objective_fn_name + ": " + str(new_value))
        if self.old_value:
            print_verb(
                self.objective_fn_name, "change: " + str(self.value - self.old_value)
            )
            self.switch_solver_maybe()
        print_verb()
        print_verb("iteration took", time.time() - start_time, "seconds")
        print_verb("\n")

    def optimise(self) -> None:
        for i in range(self.max_iter):
            self.current_iteration = i
            self.perform_iteration()

        print_verb("performing final run with optimised initial condition")
        final_inverse_value = self.run_fn(self.parameters, True)
        final_value = self.inv_fn(final_inverse_value)
        print_verb()
        print_verb(self.objective_fn_name + ":", final_value)
        print_verb(self.objective_fn_name + " change:", (final_value - self.value))
        self.old_value = self.value
        self.value = final_value
        print_verb()


class OptimiserNonFourier(Optimiser):

    def make_noisy(
        self, input: "input_type", noise_amplitude: jsd_float = 1e-1
    ) -> "input_type":
        input_ = cast(VectorField[PhysicalField], input)
        e0 = input_.energy()
        interval_bound = e0**0.5 * noise_amplitude / 2
        return VectorField(
            [
                f
                + FourierField.FromRandom(
                    input_.get_physical_domain(),
                    seed=37,
                    interval=(-interval_bound, interval_bound),
                ).no_hat()
                for f in input_
            ]
        )

    def parameters_to_run_input(self, parameters: parameter_type) -> "input_type":
        if self.parameters_to_run_input_fn == None:
            if self.force_2d:
                vort_hat = None
                v1 = parameters[0].real
                v0_00_hat = parameters[1]
                v2_00_hat = None
            else:
                vort = parameters[0].real
                v1 = parameters[1].real
                v0_00_hat = parameters[2]
                v2_00_hat = parameters[3]
                vort = self.domain.update_boundary_conditions(vort)
                vort_hat = self.domain.field_hat(vort)
            v1 = self.domain.update_boundary_conditions(v1)
            v1_hat = self.domain.field_hat(v1)
            domain = self.domain
            U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
                domain, vort_hat, v1_hat, v0_00_hat, v2_00_hat, two_d=self.force_2d
            )
            input = VectorField.FromData(FourierField, domain, U_hat_data).no_hat()
        else:
            assert self.parameters_to_run_input_fn is not None
            input = self.parameters_to_run_input_fn(parameters)

        def set_time_step_rec(inp: "input_type") -> None:
            if isinstance(inp, Field) or isinstance(inp, VectorField):
                inp.set_time_step(self.current_iteration)
            else:
                for inp_i in inp:
                    set_time_step_rec(inp_i)

        set_time_step_rec(input)
        return input

    def run_input_to_parameters(self, input: "input_type") -> parameter_type:
        input_ = cast(VectorField[PhysicalField], input)
        if self.run_input_to_parameters_fn == None:
            input_hat = input_.hat()
            if self.force_2d:
                v0_1 = input_[1].data * (1 + 0j)
                v0_0_00_hat = input_hat[0].data[0, :, 0] * (1 + 0j)
                self.parameters = tuple([v0_1, v0_0_00_hat])
            else:
                vort = input_.curl()[1].data * (1 + 0j)
                v0_1 = input_[1].data * (1 + 0j)
                v0_0_00_hat = input_hat[0].data[0, :, 0]
                v2_0_00_hat = input_hat[2].data[0, :, 0]
                self.parameters = tuple([vort, v0_1, v0_0_00_hat, v2_0_00_hat])
        else:
            assert self.run_input_to_parameters_fn is not None
            self.parameters = self.run_input_to_parameters_fn(input_)
        return self.parameters

    def post_process_iteration(self) -> None:

        i = self.current_iteration
        U = cast(
            VectorField[PhysicalField], self.parameters_to_run_input(self.parameters)
        )
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        v0_new[0].plot_isosurfaces(0.4)
        v0_new.save_to_file("vel_0_" + str(i + 1))


class OptimiserPertAndBase(Optimiser):

    def make_noisy(
        self, input: "input_type", noise_amplitude: jsd_float = 1e-1
    ) -> "input_type":
        input_ = cast(
            tuple[VectorField[FourierField], VectorField[FourierField]], input
        )
        parameters_no_hat = input_[0].no_hat()
        e0 = parameters_no_hat.energy()
        interval_bound = e0**0.5 * noise_amplitude / 2
        # only add noise to perturbation field
        return (
            VectorField(
                [
                    f
                    + FourierField.FromRandom(
                        input_[0].get_physical_domain(),
                        seed=37,
                        interval=(-interval_bound, interval_bound),
                    )
                    for f in input_[0]
                ]
            ),
            input_[1],
        )

    def post_process_iteration(self) -> None:

        i = self.current_iteration
        U_hat, U_base_hat = self.parameters_to_run_input(self.parameters)
        U = cast(VectorField[FourierField], U_hat).no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        v0_new[0].plot_isosurfaces(0.4)
        v0_new.save_to_file("vel_0_" + str(i + 1))

        U_base = cast(VectorField[FourierField], U_base_hat).no_hat()
        U_base.set_name("vel_base")
        U_base.set_time_step(i + 1)
        U_base[0].plot_3d(2)
        U_base[0].plot_center(1)
        U_base[0].save_to_file("vel_base_" + str(i + 1))

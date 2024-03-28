#!/usr/bin/env python3

import time
import jax
import jax.numpy as jnp
import pickle

from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import Field, FourierField, VectorField
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

try:
    import optax
except ModuleNotFoundError:
    print("optax not found")
try:
    import jaxopt
except ModuleNotFoundError:
    print("jaxopt not found")


class Optimiser:

    def __init__(
        self,
        domain,
        run_fn,
        run_input_initial,
        minimise=False,
        force_2d=False,
        max_iter=20,
        use_optax=False,
        min_optax_iter=0,
        **params
    ):

        self.parameters_to_run_input_fn = params.get("parameters_to_run_input_fn")
        self.run_input_to_parameters_fn = params.get("run_input_to_parameters_fn")

        self.domain = domain
        self.force_2d = force_2d
        if type(run_input_initial) is str:
            self.parameter_file_name = run_input_initial
            self.parameters = self.parameters_from_file()
        else:
            self.parameter_file_name = "parameters"
            run_input = run_input_initial
            self.parameters = self.run_input_to_parameters(run_input)
        self.old_value = None
        self.current_iteration = 0
        self.max_iter = max_iter
        self.min_optax_iter = min_optax_iter
        if minimise:
            self.inv_fn = lambda x: x
        else:
            self.inv_fn = lambda x: -x
        self.run_fn = lambda v, out=False: self.inv_fn(run_fn(self.parameters_to_run_input(v), out))
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

    def parameters_to_run_input(self, parameters):
        if self.parameters_to_run_input_fn == None:
            if self.force_2d:
                vort_hat = None
                v1_hat = parameters[0]
                v0_00 = parameters[1]
                v2_00 = None
            else:
                vort_hat = parameters[0]
                v1_hat = parameters[1]
                v0_00 = parameters[2]
                v2_00 = parameters[3]
            domain = self.domain
            U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
                domain, vort_hat, v1_hat, v0_00, v2_00, two_d=True
            )
            input = VectorField.FromData(FourierField, domain, U_hat_data)
        else:
            input = self.parameters_to_run_input_fn(parameters)
        def set_time_step_rec(inp):
            if isinstance(inp, Field) or isinstance(inp, VectorField):
                inp.set_time_step(self.current_iteration)
            else:
                [set_time_step_rec(inp_i) for inp_i in inp]
        set_time_step_rec(input)
        return input

    def parameters_from_file(self):
        """Load paramters from file filename."""
        print_verb("loading parameters from", self.parameter_file_name)
        with open(Field.field_dir + self.parameter_file_name, 'rb') as file:
            self.parameters = pickle.load(file)
        return self.parameters

    def parameters_to_file(self):
        """Save paramters to file filename."""
        with open(Field.field_dir + self.parameter_file_name, 'wb') as file:
            pickle.dump(self.parameters, file, protocol=pickle.HIGHEST_PROTOCOL)

    def run_input_to_parameters(self, input):
        if self.run_input_to_parameters_fn == None:
            if self.force_2d:
                v0_1 = input[1].data * (1 + 0j)
                v0_0_00_hat = input[0].data[0, :, 0] * (1 + 0j)
                self.parameters = tuple([v0_1, v0_0_00_hat])
            else:
                vort_hat = input.curl()[1].data * (1 + 0j)
                v0_1 = input[1].data * (1 + 0j)
                v0_0_00_hat = input[0].data[0, :, 0] * (1 + 0j)
                v2_0_00_hat = input[2].data[0, :, 0] * (1 + 0j)
                self.parameters = tuple([vort_hat, v0_1, v0_0_00_hat, v2_0_00_hat])
        else:
            self.parameters = self.run_input_to_parameters_fn(input)
        return self.parameters

    def get_parameters_norm(self):
        return jnp.linalg.norm(
            jnp.concatenate([jnp.array(v.flatten()) for v in self.parameters])
        )

    def get_optax_solver(self, learning_rate=1e-2, scale_by_norm=True):
        learning_rate_ = (
            learning_rate * self.get_parameters_norm()
            if scale_by_norm
            else learning_rate
        )
        # opt = optax.adagrad(learning_rate=learning_rate_) # minimizer
        # opt = optax.adabelief(learning_rate=learning_rate_) # minimizer
        opt = optax.adam(learning_rate=learning_rate_)  # minimizer
        solver = jaxopt.OptaxSolver(
            opt=opt, fun=jax.value_and_grad(self.run_fn), value_and_grad=True, jit=True
        )
        return solver

    def get_jaxopt_solver(self):
        # solver = jaxopt.NonlinearCG(jax.value_and_grad(self.run_fn), True, jit=False, implicit_diff=True, maxls=2) # minimizer
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

    def post_process_iteration(self):

        i = self.current_iteration
        U_hat = self.parameters_to_run_input(self.parameters)
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        cont_error = v0_new.div().energy() / v0_new.energy()
        print_verb("cont_error", cont_error)

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        self.parameters_to_file()

    def switch_solver_maybe(self):
        i = self.current_iteration
        min_number_of_optax_steps = self.min_optax_iter
        if (
            (not self.solver_switched)
            and i >= min_number_of_optax_steps
            and self.value - self.old_value < 0
        ):
            print_verb("switching to jaxopt solver")
            self.solver = self.get_jaxopt_solver()
            self.solver_switched = True
            self.state = solver.init_state(self.parameters)

    def perform_iteration(self):
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

    def optimise(self):
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

class OptimiserPertAndBase(Optimiser):

    def post_process_iteration(self):

        i = self.current_iteration
        U_hat, U_base_hat = self.parameters_to_run_input(self.parameters)
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        v0_new.save_to_file("vel_0_" + str(i + 1))

        U_base = U_base_hat.no_hat()
        U_base.set_name("vel_base")
        U_base.set_time_step(i + 1)
        U_base[0].plot_3d(2)
        U_base[0].plot_center(1)
        U_base[0].save_to_file("vel_base_" + str(i + 1))

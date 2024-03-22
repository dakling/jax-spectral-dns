#!/usr/bin/env python3

import time
import jax
import jax.numpy as jnp

from equation import print_verb
from field import FourierField, VectorField
from navier_stokes_perturbation import NavierStokesVelVortPerturbation

try:
    import optax
except ModuleNotFoundError:
    print("optax not found")
try:
    import jaxopt
except ModuleNotFoundError:
    print("jaxopt not found")

class Optimiser:

    def __init__(self, domain, run_fn, v_hat_initial, minimise=False, force_2d=False, max_iter=20, use_optax=False, min_optax_iter=0, **params):
        self.domain = domain
        self.v_hat = v_hat_initial
        self.parameters = self.vel_hat_to_parameters()
        self.value = None
        self.old_value = None
        self.current_iteration = 0
        self.max_iter = max_iter
        self.min_optax_iter = min_optax_iter
        self.force_2d = force_2d
        if minimise:
            self.inv_fn = lambda x: x
        else:
            self.inv_fn = lambda x: -x
        self.run_fn = self.inv_fn(run_fn)
        if use_optax:
            try:
                learning_rate = params["learning_rate"]
            except KeyError:
                learning_rate = 1e-2
            try:
                scale_by_norm = params["scale_by_norm"]
            except KeyError:
                scale_by_norm = True
            self.solver = self.get_optax_solver(learning_rate, scale_by_norm)
            self.solver_switched = False
        else:
            self.solver = self.get_jaxopt_solver()
            self.solver_switched = True
        self.state = self.solver.init_state(self.parameters)
        try:
            self.objective_fn_name = params["objective_fn_name"]
        except KeyError:
            self.objective_fn_name = "objective function"

    def parameters_to_vel_hat(self):
        v1_hat = self.parameters[0]
        v0_00 = self.parameters[1]
        domain = self.domain
        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(domain, None, v1_hat, v0_00, None, two_d=True)
        U_hat = VectorField.FromData(FourierField, domain, U_hat_data)
        U_hat.set_name("velocity_hat")
        U_hat.set_time_step(self.current_iteration)
        return U_hat

    def vel_hat_to_parameters(self):
        v_hat = self.v_hat
        v0_1 = v_hat[1].data * (1+0j)
        v0_0_00_hat = v_hat[0].data[0, :, 0] * (1+0j)
        self.parameters = tuple([v0_1, v0_0_00_hat])
        return self.parameters

    def get_parameters_norm(self):
        return jnp.linalg.norm(jnp.concatenate([jnp.array(v.flatten()) for v in self.parameters]))

    def get_optax_solver(self, learning_rate=1e-2, scale_by_norm=True):
        learning_rate_ = learning_rate * self.get_parameters_norm() if scale_by_norm else learning_rate
        # opt = optax.adagrad(learning_rate=learning_rate_) # minimizer
        # opt = optax.adabelief(learning_rate=learning_rate_) # minimizer
        opt = optax.adam(learning_rate=learning_rate_) # minimizer
        solver = jaxopt.OptaxSolver(opt=opt, fun=jax.value_and_grad(self.run_fn), value_and_grad=True, jit=True) # minimizer
        return solver

    def get_jaxopt_solver(self):
        # solver = jaxopt.NonlinearCG(jax.value_and_grad(self.run_fn), True, jit=False, implicit_diff=True, maxls=2) # minimizer
        solver = jaxopt.LBFGS(jax.value_and_grad(self.run_fn), value_and_grad=True, implicit_diff=True, jit=True, linesearch="zoom", linesearch_init="current", maxls=15) # minimizer
        return solver

    def post_process_iteration(self):

        i = self.current_iteration
        U_hat = self.parameters_to_vel_hat()
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()
        print_verb("relative continuity error:", v0_new.div().energy() / v0_new.energy())

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        v0_new.save_to_file("vel_0_" + str(i + 1))

    def switch_solver_maybe(self):
        i = self.current_iteration
        min_number_of_optax_steps = self.min_optax_iter
        if (not self.solver_switched) and i >= min_number_of_optax_steps and self.value - self.old_value < 0:
            print_verb("switching to jaxopt solver")
            solver = self.get_jaxopt_solver()
            self.solver_switched = True
            self.state = solver.init_state(self.parameters)

    def perform_iteration(self):
        start_time = time.time()
        i = self.current_iteration
        number_of_steps = self.max_iter
        print_verb("Iteration", i+1, "of", number_of_steps)

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
            print_verb(self.objective_fn_name, "change: " + str(self.value - self.old_value))
            self.switch_solver_maybe()
        print_verb()

        print_verb("iteration took", time.time() - start_time, "seconds")
        print_verb("\n")


    def optimize(self):
        for i in range(self.max_iter):
            self.current_iteration = i
            self.perform_iteration()

        print_verb("performing final run with optimised initial condition")
        final_inverse_value = self.run_fn(self.parameters, True)
        final_value = - final_inverse_value
        print_verb()
        print_verb(self.objective_fn_name + ":", final_value)
        print_verb(self.objective_fn_name + " change:", (final_value - self.value))
        self.old_value = self.value
        self.value = final_value
        print_verb()

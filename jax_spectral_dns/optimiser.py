#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
    cast,
)
import jax
import jax.numpy as jnp
import pickle

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import Field, FourierField, PhysicalField, VectorField
from jax_spectral_dns.linear_stability_calculation import LinearStabilityCalculation
from jax_spectral_dns.navier_stokes_perturbation import NavierStokesVelVortPerturbation

# from jax.sharding import Mesh
# from jax.sharding import PartitionSpec
# from jax.sharding import NamedSharding
# from jax.experimental import mesh_utils

# P = jax.sharding.PartitionSpec
# n = jax.local_device_count()
# devices = mesh_utils.create_device_mesh((n,))
# mesh = jax.sharding.Mesh(devices, ("x",))
# sharding = jax.sharding.NamedSharding(mesh, P("x"))  # type: ignore[no-untyped-call]

if TYPE_CHECKING:
    from jax_spectral_dns._typing import jsd_float, parameter_type, jnp_array, jsd_array

try:
    import optax  # type: ignore
except Exception:
    print("optax could not be loaded")
try:
    import jaxopt  # type: ignore
except ModuleNotFoundError:
    print("jaxopt not found")
try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    pass

I = TypeVar("I")  # the type of the input to the run-function


class Optimiser(ABC, Generic[I]):

    def __init__(
        self,
        calculation_domain: PhysicalDomain,
        optimisation_domain: PhysicalDomain,
        run_fn: Union[
            Callable[[I, bool], "jsd_float"],
            Callable[[I, bool], Tuple["jsd_float", "jsd_array"]],
        ],
        run_input_initial: Union[I, str],
        value_and_grad: bool = False,
        minimise: bool = False,
        force_2d: bool = False,
        max_iter: int = 20,
        use_optax: bool = False,
        min_optax_iter: int = 0,
        add_noise: bool = True,
        noise_amplitude: float = 1e-6,
        **params: Any,
    ):

        self.parameters_to_run_input_fn = params.get("parameters_to_run_input_fn")
        self.run_input_to_parameters_fn = params.get("run_input_to_parameters_fn")

        self.calculation_domain = calculation_domain
        self.optimisation_domain = optimisation_domain
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
                run_input: I = self.make_noisy(
                    cast(I, run_input_initial), noise_amplitude=noise_amplitude
                )
            else:
                run_input = cast(I, run_input_initial)
            self.parameters = self.run_input_to_parameters_(run_input)
        self.old_value: Optional["jsd_float"] = None
        self.current_iteration: int = 0
        self.max_iter: int = max_iter
        self.min_optax_iter: int = min_optax_iter
        if minimise:
            self.inv_fn: Callable[["jsd_float"], "jsd_float"] = lambda x: x
        else:
            self.inv_fn = lambda x: -x

        if not value_and_grad:
            run_fn_ = cast(Callable[[I, bool], "jsd_float"], run_fn)
            self.run_fn: Callable[["parameter_type", bool], "jsd_float"] = lambda v, out=False: self.inv_fn(  # type: ignore[misc]
                run_fn_(self.parameters_to_run_input_(v), out)
            )
            self.value_and_grad_fn = jax.value_and_grad(self.run_fn)
        else:
            run_fn__ = cast(
                Callable[[I, bool], Tuple["jsd_float", "jnp_array"]], run_fn
            )
            self.run_fn = lambda v, out=False: self.inv_fn(  # type: ignore[misc]
                run_fn__(self.parameters_to_run_input_(v), out)[0]
            )

            def vg_fn(
                v: "parameter_type", out: bool = False
            ) -> Tuple["jsd_float", "jnp_array"]:
                run_fn_ = cast(
                    Callable[[I, bool], Tuple["jsd_float", "jnp_array"]], run_fn
                )
                obj, grad = run_fn_(self.parameters_to_run_input_(v), out)
                return self.inv_fn(obj), grad

            self.value_and_grad_fn = vg_fn
        self.objective_fn_name = params.get("objective_fn_name", "objective function")
        if max_iter > 0:
            if use_optax:
                learning_rate = params.get("learning_rate", 1e-2)
                scale_by_norm = params.get("scale_by_norm", True)
                self.solver = self.get_optax_solver(learning_rate, scale_by_norm)
                self.solver_switched = False
                print_verb("Using Optax solver")
            else:
                # assert (
                #     jax.local_device_count() == 1
                # ), "jaxopt does not support multiple devices - set device_count to 1 or use optax solver."
                self.solver = self.get_jaxopt_solver()
                self.solver_switched = True
                print_verb("Using jaxopt solver")
            self.state = self.solver.init(self.parameters)
            self.value = self.inv_fn(self.state.value)
            print_verb(self.objective_fn_name + ":", self.value)

    def parameters_to_run_input_(self, parameters: "parameter_type") -> I:
        if self.parameters_to_run_input_fn == None:
            input = self.parameters_to_run_input(parameters)
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

    def run_input_to_parameters_(self, input: I) -> "parameter_type":
        if self.run_input_to_parameters_fn == None:
            self.parameters = self.run_input_to_parameters(input)
        else:
            assert self.run_input_to_parameters_fn is not None
            self.parameters = self.run_input_to_parameters_fn(input)
        return self.parameters

    @abstractmethod
    def parameters_to_run_input(self, parameters: "parameter_type") -> I: ...

    @abstractmethod
    def run_input_to_parameters(self, input: I) -> "parameter_type": ...

    @abstractmethod
    def make_noisy(self, input: I, noise_amplitude: float = 1e-1) -> I: ...

    def parameters_from_file(self) -> "parameter_type":
        """Load paramters from file filename."""
        print_verb("loading parameters from", self.parameter_file_name)
        with open(Field.field_dir + self.parameter_file_name, "rb") as file:
            self.parameters = pickle.load(file)
        return self.parameters

    def parameters_to_file(self) -> None:
        """Save paramters to file filename."""
        with open(Field.field_dir + self.parameter_file_name, "wb") as file:
            pickle.dump(self.parameters, file, protocol=pickle.HIGHEST_PROTOCOL)

    def get_parameters_norm(self) -> "jsd_float":
        return jnp.linalg.norm(
            jnp.concatenate([jnp.array(v.flatten()) for v in self.parameters])
        )

    def get_optax_solver(
        self, learning_rate: "jsd_float" = 1e-2, scale_by_norm: bool = True
    ) -> jaxopt.OptaxSolver:
        learning_rate_ = (
            learning_rate * self.get_parameters_norm()
            if scale_by_norm
            else learning_rate
        )
        # opt = optax.adam(learning_rate=learning_rate_)  # minimizer
        # opt = optax.adagrad(learning_rate=learning_rate_)  # minimizer
        opt = optax.lbfgs(learning_rate=learning_rate_)  # minimizer
        solver = jaxopt.OptaxSolver(
            opt=opt, fun=self.value_and_grad_fn, value_and_grad=True, jit=True
        )
        self.value_and_grad = optax.value_and_grad_from_state(self.run_fn)
        return solver

    def get_jaxopt_solver(self) -> jaxopt.LBFGS:
        solver = jaxopt.LBFGS(
            self.value_and_grad_fn,
            value_and_grad=True,
            implicit_diff=True,
            jit=True,
            linesearch="zoom",
            linesearch_init="current",
            maxls=15,
        )
        return solver

    @abstractmethod
    def post_process_iteration(self) -> None: ...

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
        print_verb("Iteration", i + 1, "of", number_of_steps, notify=True)

        solver = self.solver

        self.post_process_iteration()

        value, grad = self.value_and_grad(self.parameters, state=self.state)
        # self.parameters, self.state = solver.update(
        #     self.parameters, self.state, value=value, grad=grad
        # )
        updates, self.state = solver.update(
            grad,
            self.state,
            self.parameters,
            value=value,
            grad=grad,
            value_fn=self.run_fn,
        )
        self.parameters = optax.apply_updates(self.parameters, updates)
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
        iteration_duration = time.time() - start_time
        try:
            print_verb("iteration took", format_timespan(iteration_duration))
        except Exception:
            print_verb("iteration took", iteration_duration, "seconds")
        print_verb("\n")
        jax.clear_caches()  # type: ignore[no-untyped-call]

    def optimise(self) -> None:
        for i in range(self.max_iter):
            self.current_iteration = i
            self.perform_iteration()

        print_verb("performing final run with optimised initial condition")
        final_inverse_value = self.run_fn(self.parameters, True)
        final_value = self.inv_fn(final_inverse_value)
        print_verb()
        print_verb(self.objective_fn_name + ":", final_value)
        try:  # this might not work, e.g. if self.value does not exist (if max_iter is < 1). Anyway, printing should not cause an exception, so we ignore it.
            print_verb(self.objective_fn_name + " change:", (final_value - self.value))
            self.old_value = self.value
            self.value = final_value
        except Exception:
            pass
        print_verb()


class OptimiserFourier(Optimiser[VectorField[FourierField]]):

    def make_noisy(
        self, input: VectorField[FourierField], noise_amplitude: float = 1e-6
    ) -> VectorField[FourierField]:
        print_verb("adding noise")
        input = input.project_onto_domain(self.optimisation_domain)

        def get_white_noise_field(field: FourierField) -> FourierField:
            return FourierField.FromWhiteNoise(
                self.optimisation_domain,
                energy_norm=field.no_hat().energy() * noise_amplitude,
            )

        return VectorField([f + get_white_noise_field(f) for f in input])

    def parameters_to_run_input(
        self, parameters: "parameter_type"
    ) -> VectorField[FourierField]:
        if self.force_2d:
            vort_hat: Optional["jnp_array"] = None
            v1_hat: "jnp_array" = parameters[0]
            v0_00: "jnp_array" = parameters[1]
            v2_00: Optional["jnp_array"] = None
        else:
            vort_hat = parameters[0]
            v1_hat = parameters[1]
            v0_00 = parameters[2]
            v2_00 = parameters[3]
        domain = self.optimisation_domain

        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
            domain, vort_hat, v1_hat, v0_00, v2_00, two_d=self.force_2d
        )

        input: VectorField[FourierField] = VectorField.FromData(
            FourierField, domain, U_hat_data
        ).project_onto_domain(self.calculation_domain)

        return input

    def run_input_to_parameters(
        self, input: VectorField[FourierField]
    ) -> "parameter_type":

        input = input.project_onto_domain(self.optimisation_domain)
        if self.force_2d:
            v0_1 = input[1].data * (1 + 0j)
            v0_0_00_hat = input[0].data[0, :, 0] * (1 + 0j)
            self.parameters = tuple([v0_1, v0_0_00_hat])
        else:
            vort_hat = input.curl()[1].data * (1 + 0j)
            v0_1 = input[1].data * (1 + 0j)
            v0_0_00_hat = input[0].data[0, :, 0] * (1 + 0j)
            v2_0_00_hat = input[2].data[0, :, 0] * (1 + 0j)
            self.parameters = (vort_hat, v0_1, v0_0_00_hat, v2_0_00_hat)
        return self.parameters

    def post_process_iteration(self) -> None:

        self.parameters_to_file()
        i = self.current_iteration
        U_hat = self.parameters_to_run_input_(self.parameters)
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_div = v0_new.div()
        cont_error = v0_div.energy() / v0_new.energy()
        print_verb("cont_error", cont_error)
        if cont_error > 1e-1:
            v0_div.plot_3d()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.plot_3d(0)
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        try:
            v0_new[0].plot_isosurfaces(0.4)
        except Exception:
            pass


class OptimiserNonFourier(Optimiser[VectorField[PhysicalField]]):

    def make_noisy(
        self, input: "VectorField[PhysicalField]", noise_amplitude: float = 1e-6
    ) -> "VectorField[PhysicalField]":
        print_verb("adding noise")
        e0 = input.energy()
        return VectorField(
            [
                f
                + FourierField.FromRandom(
                    input.get_physical_domain(), seed=37, energy_norm=e0
                ).no_hat()
                for f in input
            ]
        )

    def parameters_to_run_input(
        self, parameters: "parameter_type"
    ) -> "VectorField[PhysicalField]":
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
                vort = self.optimisation_domain.update_boundary_conditions(vort)
                vort_hat = self.optimisation_domain.field_hat(vort)
            v1 = self.optimisation_domain.update_boundary_conditions(v1)
            v1_hat = self.optimisation_domain.field_hat(v1)
            domain = self.optimisation_domain
            U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
                domain, vort_hat, v1_hat, v0_00_hat, v2_00_hat, two_d=self.force_2d
            )

            input = (
                VectorField.FromData(FourierField, domain, U_hat_data)
                .project_onto_domain(self.calculation_domain)
                .no_hat()
            )
        else:
            assert self.parameters_to_run_input_fn is not None
            input = self.parameters_to_run_input_fn(parameters)

        def set_time_step_rec(inp: "VectorField[PhysicalField]") -> None:
            if isinstance(inp, Field) or isinstance(inp, VectorField):
                inp.set_time_step(self.current_iteration)
            else:
                for inp_i in inp:
                    set_time_step_rec(inp_i)

        set_time_step_rec(input)
        return input

    def run_input_to_parameters(
        self, input: "VectorField[PhysicalField]"
    ) -> "parameter_type":
        if self.run_input_to_parameters_fn == None:
            input_hat = input.hat()
            if self.force_2d:
                v0_1 = input[1].data * (1 + 0j)
                v0_0_00_hat = input_hat[0].data[0, :, 0] * (1 + 0j)
                self.parameters = tuple([v0_1, v0_0_00_hat])
            else:
                vort = input.curl()[1].data * (1 + 0j)
                v0_1 = input[1].data * (1 + 0j)
                v0_0_00_hat = input_hat[0].data[0, :, 0]
                v2_0_00_hat = input_hat[2].data[0, :, 0]
                self.parameters = tuple([vort, v0_1, v0_0_00_hat, v2_0_00_hat])
        else:
            assert self.run_input_to_parameters_fn is not None
            self.parameters = self.run_input_to_parameters_fn(input)
        return self.parameters

    def post_process_iteration(self) -> None:

        i = self.current_iteration
        U = self.parameters_to_run_input_(self.parameters)
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.save_to_file("vel_0_" + str(i + 1))
        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        try:
            v0_new[0].plot_isosurfaces(0.4)
        except Exception:
            pass


class OptimiserPertAndBase(
    Optimiser[Tuple[VectorField[FourierField], VectorField[FourierField]]]
):

    def make_noisy(
        self,
        input: "Tuple[VectorField[FourierField], VectorField[FourierField]]",
        noise_amplitude: float = 1e-6,
    ) -> "Tuple[VectorField[FourierField], VectorField[FourierField]]":
        parameters_no_hat = input[0].no_hat()
        e0 = parameters_no_hat.energy()
        # only add noise to perturbation field
        return (
            VectorField(
                [
                    f
                    + FourierField.FromRandom(
                        input[0].get_physical_domain(), seed=37, energy_norm=e0
                    )
                    for f in input[0]
                ]
            ),
            input[1],
        )

    def post_process_iteration(self) -> None:

        i = self.current_iteration
        U_hat, U_base_hat = self.parameters_to_run_input_(self.parameters)
        U = U_hat.no_hat()
        U.update_boundary_conditions()
        v0_new = U.normalize_by_energy()

        v0_new.set_name("vel_0")
        v0_new.set_time_step(i + 1)
        v0_new.save_to_file("vel_0_" + str(i + 1))

        U_base = U_base_hat.no_hat()
        U_base.set_name("vel_base")
        U_base.set_time_step(i + 1)
        U_base[0].save_to_file("vel_base_" + str(i + 1))

        v0_new.plot_3d(2)
        v0_new[0].plot_center(1)
        v0_new[1].plot_center(1)
        try:
            v0_new[0].plot_isosurfaces(0.4)
        except Exception:
            pass

        U_base[0].plot_3d(2)
        U_base[0].plot_center(1)

    def run_input_to_parameters(
        self, inp: "Tuple[VectorField[FourierField], VectorField[FourierField]]"
    ) -> "parameter_type":
        vel_hat, vel_base = inp
        v0_1 = vel_hat[1].data[1, :, 0] * (1 + 0j)
        v0_0_00_hat = vel_hat[0].data[0, :, 0] * (1 + 0j)

        # optimise entire slice
        # v0_base_hat = vel_base[0].get_data()[0, :, 0]

        # optimise using phi_s basis
        # v0_base_hat_coeffs = jnp.array([-0.5+0j, 0.0+0j, 0.0+0j])

        # optimise using parametric profile
        v0_base_hat_coeffs = jnp.array([jnp.log(2.0 + 0j), jnp.log(1.0 + 0j)])

        v0 = tuple([v0_1, v0_0_00_hat, v0_base_hat_coeffs])
        return v0

    def parameters_to_run_input(
        self, parameters: "parameter_type"
    ) -> Tuple[VectorField[FourierField], VectorField[FourierField]]:
        domain = self.optimisation_domain
        if self.force_2d:
            vort_hat: Optional["jnp_array"] = None
            v1_hat: "jnp_array" = parameters[0][0]
            v0_00: "jnp_array" = parameters[0][1]
            v2_00: Optional["jnp_array"] = None
        else:
            vort_hat = parameters[0][0]
            v1_hat = parameters[0][1]
            v0_00 = parameters[0][2]
            v2_00 = parameters[0][3]
        U_hat_data = NavierStokesVelVortPerturbation.vort_yvel_to_vel(
            domain, vort_hat, v1_hat, v0_00, v2_00, two_d=self.force_2d
        )

        # optimise entire slice
        # v0_base_yslice = params[2]
        # v0_base_hat = domain.field_hat(lsc0.y_slice_to_3d_field(domain, v0_base_yslice))

        # optimise using phi_s basis
        lsc0 = LinearStabilityCalculation(
            Re=1, alpha=0, beta=0, n=domain.number_of_cells(1)
        )
        lsc0.symm = True
        v0_base_yslice_coeffs = parameters[2]
        v0_base_zeros = jnp.zeros_like(v0_base_yslice_coeffs)
        for _ in range(3):
            v0_base_yslice_coeffs = jnp.concatenate(
                (v0_base_yslice_coeffs, v0_base_zeros)
            )
        v0_base_hat = (
            (lsc0.velocity_field(domain, v0_base_yslice_coeffs)).hat()[0].get_data()
        )

        # # optimise using parametric profile
        # v0_base_yslice = parameters[2]
        # m = jnp.exp(v0_base_yslice_coeffs[0].real)  # ensures > 0
        # n = jnp.exp(v0_base_yslice_coeffs[1].real)  # ensures > 0
        # print_verb("m, n:", m, n)
        # v0_base_yslice = jnp.array(
        #     list(map(lambda y: (1 - y ** (m)) ** (1 / n), domain.grid[1]))
        # )
        # v0_base_hat = domain.field_hat(lsc0.y_slice_to_3d_field(domain, v0_base_yslice))

        U_hat: VectorField[FourierField] = VectorField.FromData(
            FourierField, domain, U_hat_data
        ).project_onto_domain(self.calculation_domain)
        U_base: VectorField[FourierField] = VectorField.FromData(
            FourierField,
            domain,
            jnp.array(
                [
                    v0_base_hat,
                    jnp.zeros_like(v0_base_hat),
                    jnp.zeros_like(v0_base_hat),
                ]
            ),
        ).project_onto_domain(self.calculation_domain)
        return (U_hat, U_base)

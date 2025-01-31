#!/usr/bin/env python3
from __future__ import annotations

from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    TYPE_CHECKING,
    cast,
)
import time

import jax
import jax.numpy as jnp
import numpy as np

from pathlib import Path
import os
import h5py  # type: ignore

try:
    import gi  # type: ignore

    gi.require_version("Notify", "0.7")
    from gi.repository import Notify  # type: ignore

    Notify.init("jax-spectral-dns")
except Exception:
    print("gi package not found, notifications will not work.")

try:
    from humanfriendly import format_timespan  # type: ignore
except ModuleNotFoundError:
    print("humanfriendly not found, time durations won't be formatted nicely.")


from jax_spectral_dns.domain import Domain, PhysicalDomain
from jax_spectral_dns.field import Field, FourierField, PhysicalField, VectorField
from jax_spectral_dns.fixed_parameters import FixedParameters

if TYPE_CHECKING:
    from jax_spectral_dns._typing import AnyField, AnyFieldList, jsd_float


jnp_int_array = jnp.ndarray

NoneType = type(None)


def print_verb(
    *in_str: Any, verbosity_level: int = 1, debug: bool = False, notify: bool = False
) -> None:
    pref = "[" + time.ctime() + "]  " + "  " * verbosity_level
    if Equation.verbosity_level >= verbosity_level:
        if debug:
            print(pref, end=" ")
            for st in in_str:
                if type(st) is str:
                    print(st, end=" ")
                else:
                    jax.debug.callback(lambda x: print(x, end=" "), st)
            print()
        else:
            if notify:
                try:
                    Notify.Notification.new(*in_str).show()
                except Exception:
                    pass
            print(pref, *in_str)


E = TypeVar("E", bound="Equation")


class Equation:
    name = "equation"
    write_intermediate_output = False
    write_entire_output = False

    # verbosity_level:
    # 0: no output
    # 1: mostly output from examples.py informing the user about what is being done
    # 2: additional helpful output from the solver
    # 3: even more additional output from the solver that is usually nonessential
    verbosity_level: int = 1

    def __init__(self: E, domain: Domain, *fields: "AnyField", **params: Any):
        dt: "float" = params.get("dt", 1e-2)
        self.fixed_parameters = FixedParameters(domain, dt)
        self.fields = {}
        self.time_step: int = 0
        self.time: "jsd_float" = 0.0
        self.before_time_step_fn: Optional[Callable[[E], None]] = None
        self.after_time_step_fn: Optional[Callable[[E], None]] = None
        self.post_process_fn: Optional[Callable[[E, int], None]] = None
        self.end_time = params.get("end_time", 0.0)
        self.max_iter = params.get("max_iter", 1000)
        for field in fields:
            f_name: str = field.get_name()
            self.fields[f_name] = [field]
            self.fields[f_name][0].set_name(f_name + "_0")
        # self.initialize()

    @classmethod
    def initialize(cls, cleanup: bool = True) -> None:
        Field.initialize(cleanup)

    @classmethod
    def find_suitable_dt(
        cls,
        domain: PhysicalDomain,
        max_cfl: float = 0.7,
        U_max: tuple[float, ...] = (1.0, 1e-10, 1e-10),
        end_time: Optional[float] = None,
        safety_factor: float = 0.9,
    ) -> float:
        """Returns a suitable time step based on the given CFL number. If
        end_time is provided, it is assumed that a number of time steps allowing
        for a favourable partition into inner and outer steps (relevant for
        solving with solve_scan) is desired, and this is taken into account."""
        dT = [
            max_cfl
            * (
                domain.get_extent(i) / domain.number_of_cells(i)
                if domain.is_periodic(i)
                else 1.0
            )
            / U_max[i]
            for i in domain.all_dimensions()
        ]
        dt = safety_factor * min(dT)
        if end_time is None:
            return dt

        def median_factor(n: int) -> int:
            """Return the median integer factor of n."""
            factors = reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
            factors.sort()
            number_of_factors = len(factors)  # should always be divisible by 2
            return factors[number_of_factors // 2]

        number_of_time_steps = max(0, int(end_time / dt))
        number_of_inner_steps = median_factor(number_of_time_steps)
        number_of_outer_steps = number_of_time_steps // number_of_inner_steps
        bad_n_step_division = (
            abs(np.sqrt(number_of_time_steps)) - number_of_outer_steps
            > number_of_outer_steps
        )
        while bad_n_step_division:
            number_of_time_steps += 1
            number_of_inner_steps = median_factor(number_of_time_steps)
            number_of_outer_steps = number_of_time_steps // number_of_inner_steps
            bad_n_step_division = (
                abs(np.sqrt(number_of_time_steps)) - number_of_outer_steps
                > number_of_outer_steps
            )

        dt = end_time / number_of_time_steps
        return dt

    def get_dt(self) -> "float":
        return self.fixed_parameters.dt

    def get_domain(self) -> Domain:
        return self.fixed_parameters.domain

    def get_physical_domain(self) -> PhysicalDomain:
        raise NotImplementedError

    def get_field(self, name: str, index: int) -> "AnyField":
        try:
            out: "AnyField" = self.fields[name][index]
            return out
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def get_fields(self, name: str) -> "AnyFieldList":
        try:
            return cast("AnyFieldList", self.fields[name])
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def get_initial_field(self, name: str) -> "AnyField":
        out = self.get_field(name, 0)
        return out

    def get_latest_field(self, name: str) -> "AnyField":
        out = self.get_field(name, -1)
        return out

    def get_number_of_fields(self, name: str) -> int:
        return len(self.get_fields(name))

    def set_field(self, name: str, index: int, field: "AnyField") -> None:
        try:
            self.fields[name][index] = field
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def set_initial_field(self, name: str, field: "AnyField") -> None:
        self.clear_field("velocity_hat")
        try:
            self.append_field(name, field)
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def append_field(self, name: str, field: "AnyField", in_place: bool = True) -> None:
        try:
            if in_place and len(self.fields[name]) > 0:
                self.fields[name][-1] = field
            else:
                self.fields[name].append(field)
            self.fields[name][-1].name = name
        except KeyError:
            raise KeyError("Expected field named " + name + " in " + self.name + ".")

    def add_field(self, name: str, field: Optional["AnyField"] = None) -> None:
        assert name not in self.fields, "Field " + name + " already exists!"
        if type(field) == NoneType:
            self.fields[name] = []
        else:
            # self.fields[name] = jnp.array([field]) # TODO avoid the use of lists
            assert field is not None
            self.fields[name] = [field]
            self.fields[name][0].name = name + "_0"

    def add_field_history(self, name: str, field_history: "AnyFieldList") -> None:
        assert name not in self.fields, "Field " + name + " already exists!"
        self.fields[name] = cast(List["AnyField"], field_history)

    def clear_field(self, name: str) -> None:
        self.fields[name] = []

    def activate_jit(self) -> None:
        Field.activate_jit_ = True

    def deactivate_jit(self) -> None:
        Field.activate_jit_ = False

    def all_dimensions_jnp(self) -> jnp_int_array:
        return jnp.arange(self.get_domain().number_of_dimensions)

    def all_dimensions(self) -> Sequence[int]:
        return range(self.get_domain().number_of_dimensions)

    def all_periodic_dimensions(self) -> list[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if self.get_domain().periodic_directions[d]
        ]

    def all_nonperiodic_dimensions(self) -> list[int]:
        return [
            self.all_dimensions()[d]
            for d in self.all_dimensions()
            if not self.get_domain().periodic_directions[d]
        ]

    def all_periodic_dimensions_jnp(self) -> jnp_int_array:
        return jnp.array(
            [
                self.all_dimensions_jnp()[d]
                for d in self.all_dimensions_jnp()
                if self.get_domain().periodic_directions[d]
            ]
        )

    def all_nonperiodic_dimensions_jnp(self) -> jnp_int_array:
        return jnp.array(
            [
                self.all_dimensions_jnp()[d]
                for d in self.all_dimensions_jnp()
                if not self.get_domain().periodic_directions[d]
            ]
        )

    def done(self) -> bool:
        iteration_done = False
        time_done = False
        if type(self.max_iter) != NoneType:
            iteration_done = self.time_step > self.max_iter
        if type(self.end_time) != NoneType:
            time_done = self.time >= self.end_time + self.get_dt() + 1e-40
        return iteration_done or time_done

    def perform_time_step(
        self,
        _: Optional[Any] = None,
        __: Optional[Any] = None,
        time_step: Optional[int] = None,
    ) -> Any:
        raise NotImplementedError()

    def set_before_time_step_fn(self: E, fn: Optional[Callable[[E], None]]) -> None:
        self.before_time_step_fn = fn  # type: ignore[assignment]

    def set_after_time_step_fn(self: E, fn: Optional[Callable[[E], None]]) -> None:
        self.after_time_step_fn = fn  # type: ignore[assignment]

    def set_post_process_fn(self: E, fn: Optional[Callable[[E, int], None]]) -> None:
        self.post_process_fn = fn  # type: ignore[assignment]

    def before_time_step(self: E) -> None:
        if type(self.before_time_step_fn) != NoneType:
            assert self.before_time_step_fn is not None
            self.before_time_step_fn(self)

    def after_time_step(self: E) -> None:
        if type(self.after_time_step_fn) != NoneType:
            assert self.after_time_step_fn is not None
            self.after_time_step_fn(self)

    def post_process(self: E) -> None:
        if type(self.post_process_fn) != NoneType:
            raise NotImplementedError()

    def prepare(self) -> None:
        pass

    def update_time(self) -> None:
        self.time += self.get_dt()
        self.time_step += 1
        # for _, field in self.fields.items():
        #     field[-1].time_step = self.time_step

    # @partial(jax.jit, static_argnums=(0))
    def solve_scan(self) -> Any:
        raise NotImplementedError()

    def solve(self) -> Any:
        self.prepare()

        if Field.activate_jit_:
            msg = (
                "Solving using jit/scan - this offers high performance but "
                "intermediate results won't be available until after the "
                "calculation finishes. To disable "
                "high-performance mode, use the deactivate_jit()-method of the "
                "Equation class."
            )

            print_verb(msg, verbosity_level=2)
            start_time = time.time()
            trajectory, _, number_of_time_steps = self.solve_scan()
            velocity_final = VectorField(
                [
                    FourierField(
                        self.get_physical_domain(),
                        trajectory[-1][i],
                        name="velocity_hat_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            velocity_final.set_time_step(len(trajectory) - 1)
            self.append_field("velocity_hat", velocity_final, in_place=False)

            print("error (eq):", jnp.linalg.norm(trajectory[0] - trajectory[1]))
            if os.environ.get("JAX_SPECTRAL_DNS_FIELD_DIR") is not None:
                try:
                    print_verb("writing trajectory to file...", verbosity_level=2)

                    with h5py.File(Field.field_dir + "/trajectory", "w") as f:
                        f.create_dataset(
                            "trajectory",
                            data=trajectory,
                            compression="gzip",
                            compression_opts=9,
                        )
                    print_verb("done writing trajectory to file", verbosity_level=2)
                except Exception:
                    print_verb("writing trajectory to file failed", verbosity_level=2)
            else:
                print_verb("not writing trajectory to file", verbosity_level=2)

            duration = time.time() - start_time
            try:
                print_verb(
                    "Took "
                    + format_timespan(duration)
                    + " for "
                    + str(number_of_time_steps)
                    + " time steps ("
                    + "{:.3e}".format(duration / number_of_time_steps)
                    + " s/TS).",
                    verbosity_level=1,
                )
            except Exception:
                print_verb(
                    "Took "
                    + "{:.2f}".format(duration)
                    + " seconds for "
                    + str(number_of_time_steps)
                    + " time steps ("
                    + "{:.3e}".format(duration / number_of_time_steps)
                    + " s/TS).",
                    verbosity_level=1,
                )
            self.deactivate_jit()
        else:
            msg = (
                "WARNING: Solving without jit/scan - performance will be "
                "significantly lower but intermediate results will be available "
                "for printing and plotting. Only recommended for testing. "
                "To enable high-performance mode, use the "
                "activate_jit()-method of the Equation class."
            )
            print_verb(msg, verbosity_level=0)
            while not self.done():
                i = self.time_step
                print_verb(
                    "Time Step "
                    + str(i + 1)
                    + ", time: "
                    + str(self.time)
                    + ", dt: "
                    + str(self.get_dt())
                )
                start_time = time.time()
                self.before_time_step()
                self.perform_time_step(None, None, i)
                self.update_time()
                self.after_time_step()
                print_verb("Took " + str(time.time() - start_time) + " seconds")

    def plot(self) -> None:
        raise NotImplementedError()

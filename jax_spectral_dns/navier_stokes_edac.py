#!/usr/bin/env python3
from __future__ import annotations

NoneType = type(None)
from operator import rshift
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union, cast
from typing_extensions import Self
import jax
import jax.numpy as jnp
from matplotlib import axes
import numpy as np
from functools import partial, reduce
import matplotlib.figure as figure
from matplotlib.axes import Axes

# from importlib import reload
import sys

from jax_spectral_dns.domain import PhysicalDomain, FourierDomain
from jax_spectral_dns.field import (
    Field,
    PhysicalField,
    VectorField,
    FourierField,
    FourierFieldSlice,
)
from jax_spectral_dns.equation import Equation, E, print_verb
from jax_spectral_dns.fixed_parameters import NavierStokesVelVortFixedParameters
from jax_spectral_dns.linear_stability_calculation import LinearStabilityCalculation

if TYPE_CHECKING:
    from jax_spectral_dns._typing import (
        np_float_array,
        np_complex_array,
        jsd_float,
        jnp_array,
        np_jnp_array,
        Vel_fn_type,
    )


class NavierStokesEDAC(Equation):
    name = "Navier Stokes equation (pseudo-compressible EDAC formulation)"

    def __init__(self, velocity_field: VectorField[PhysicalField], **params: Any):
        domain = velocity_field.get_physical_domain()

        u_max_over_u_tau = params.get("u_max_over_u_tau", 1.0)

        dt = params.get("dt", 1e-2)

        self.end_time = params.get("end_time", -1.0)

        self.max_cfl = params.get("max_cfl", 0.7)

        self.Mach = params.get("Mach", 0.3)

        self.max_number_of_outer_steps = params.get("max_number_of_outer_steps", 100)

        try:
            self.Re_tau = params["Re_tau"]
        except KeyError:
            try:
                self.Re_tau = params["Re"] / u_max_over_u_tau
            except KeyError:
                raise Exception("Either Re or Re_tau has to be given as a parameter.")

        self.flow_rate = self.get_flow_rate(velocity_field)
        self.dPdx = -self.flow_rate * 3 / 2 / self.get_Re_tau()
        pressure_field = PhysicalField.FromFunc(
            # domain, lambda X: -self.dPdx * X[0], name="pressure"
            domain,
            lambda X: 0 * X[0],
            name="pressure",
        )

        super().__init__(domain, velocity_field, pressure_field, dt=dt)

        self.update_flow_rate()
        print_verb("calculated flow rate: ", self.flow_rate, verbosity_level=3)

    def get_domain(self) -> PhysicalDomain:
        out: PhysicalDomain = super().get_domain()  # type: ignore[assignment]
        return out

    def get_physical_domain(self) -> PhysicalDomain:
        return self.get_domain()

    def get_field(self, name: str, index: int) -> "VectorField[PhysicalField]":
        out = cast(VectorField[PhysicalField], super().get_field(name, index))
        return out

    def get_fields(self, name: str) -> List["VectorField[PhysicalField]"]:
        return cast(List[VectorField[PhysicalField]], super().get_fields(name))

    def get_initial_field(self, name: str) -> "VectorField[PhysicalField]":
        out = cast(VectorField[PhysicalField], super().get_initial_field(name))
        return out

    def get_latest_field(self, name: str) -> "VectorField[PhysicalField]":
        out = cast(VectorField[PhysicalField], super().get_latest_field(name))
        return out

    def get_flow_rate(self, vel: VectorField[PhysicalField]) -> "jsd_float":
        vel_0: PhysicalField = vel[0]
        int: PhysicalField = vel_0.definite_integral(1)  # type: ignore[assignment]
        return cast("jsd_float", int[0, 0])

    def update_flow_rate(self) -> None:
        self.flow_rate = self.get_flow_rate(self.get_latest_field("velocity"))
        self.dPdx = -self.flow_rate * 3 / 2 / self.get_Re_tau()
        self.dpdx = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: self.dPdx + 0.0 * X[0] * X[1] * X[2]
        )
        self.dpdz = PhysicalField.FromFunc(
            self.get_physical_domain(), lambda X: 0.0 + 0.0 * X[0] * X[1] * X[2]
        )

    def get_cfl(self, i: int = -1) -> "jnp_array":
        dX = (
            self.get_physical_domain().grid[0][1:]
            - self.get_physical_domain().grid[0][:-1]
        )
        dY = (
            self.get_physical_domain().grid[1][1:]
            - self.get_physical_domain().grid[1][:-1]
        )
        dZ = (
            self.get_physical_domain().grid[2][1:]
            - self.get_physical_domain().grid[2][:-1]
        )
        DX, DY, DZ = jnp.meshgrid(dX, dY, dZ, indexing="ij")
        vel = self.get_field("velocity", i)
        U = vel[0][1:, 1:, 1:]
        V = vel[1][1:, 1:, 1:]
        W = vel[2][1:, 1:, 1:]
        u_cfl = cast(float, (abs(DX) / abs(U)).min().real)
        v_cfl = cast(float, (abs(DY) / abs(V)).min().real)
        w_cfl = cast(float, (abs(DZ) / abs(W)).min().real)
        return self.get_dt() / jnp.array([u_cfl, v_cfl, w_cfl])

    def get_dt(self) -> "float":
        return self.fixed_parameters.dt

    def get_Re_tau(self) -> "jsd_float":
        return self.Re_tau

    def get_max_cfl(self) -> "jsd_float":
        return self.max_cfl

    def perform_explicit_euler_step(self, U: jnp_array, i: int) -> jnp_array:
        Re_tau = self.get_Re_tau()
        domain = self.get_domain()
        dt = self.get_dt()
        u = U[0:3]
        p = U[3]

        vort = domain.curl(u)

        vel_sq = jnp.zeros_like(u[0])
        for j in domain.all_dimensions():
            vel_sq += u[j] * u[j]
        vel_sq_nabla = []
        for i in domain.all_dimensions():
            vel_sq_nabla.append(domain.diff(vel_sq, i))

        vel_vort = domain.cross_product(u, vort)

        hel = vel_vort - 1 / 2 * jnp.array(vel_sq_nabla)

        conv_ns = -hel
        u_eq = (
            conv_ns
            - jnp.array(
                [
                    domain.diff(p, 0) + self.dpdx.data,
                    domain.diff(p, 1),
                    domain.diff(p, 2),
                ]
            )
            + 1 / Re_tau * jnp.array([domain.laplacian(u[i]) for i in range(3)])
        )

        conv_p = -(
            u[0] * domain.diff(p, 0)
            + u[1] * domain.diff(p, 1)
            + u[2] * domain.diff(p, 2)
        )

        conv_p = domain.update_boundary_conditions(conv_p)
        diff_p = 1 / Re_tau * domain.laplacian(p)
        diff_p = domain.update_boundary_conditions(diff_p)
        p_eq = (
            conv_p
            # - 1 / self.Mach**2 * (domain.diff(u[0], 0) + domain.diff(u[2], 2)) # u[1]_y is zero
            - 1 / self.Mach**2 * domain.divergence(u)
            + diff_p
        )
        new_u = u + u_eq * dt
        new_p = p + p_eq * dt
        new_u_bc = jnp.array(
            [domain.update_boundary_conditions(new_u[i]) for i in range(3)]
        )
        new_p_bc = new_p.at[0, 0, 0].set(0.0)
        return jnp.concatenate([new_u_bc, jnp.array([new_p_bc])], axis=0)

    def perform_rk_step(self, U: jnp_array, i: int) -> jnp_array:
        c = [0.0, 0.5, 0.5, 1.0]
        b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        a = [[0.0], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]]

        n_substeps = len(c)
        Re_tau = self.get_Re_tau()
        domain = self.get_domain()
        dt = self.get_dt()

        k = jnp.zeros_like(U)
        ks = [k]
        for step in range(n_substeps):
            U_ = U
            for i in range(step):
                U_ += dt * a[step][i] * ks[i]

            u = U_[0:3]
            p = U_[3]

            vort = domain.curl(u)

            vel_sq = jnp.zeros_like(u[0])
            for j in domain.all_dimensions():
                vel_sq += u[j] * u[j]
            vel_sq_nabla = []
            for i in domain.all_dimensions():
                vel_sq_nabla.append(domain.diff(vel_sq, i))

            vel_vort = domain.cross_product(u, vort)

            hel = vel_vort - 1 / 2 * jnp.array(vel_sq_nabla)

            conv_ns = -hel
            u_eq = (
                conv_ns
                - jnp.array(
                    [
                        domain.diff(p, 0) + self.dpdx.data,
                        domain.diff(p, 1),
                        domain.diff(p, 2),
                    ]
                )
                + 1 / Re_tau * jnp.array([domain.laplacian(u[i]) for i in range(3)])
            )

            conv_p = -(
                u[0] * domain.diff(p, 0)
                + u[1] * domain.diff(p, 1)
                + u[2] * domain.diff(p, 2)
            )

            conv_p = domain.update_boundary_conditions(conv_p)
            diff_p = 1 / Re_tau * domain.laplacian(p)
            diff_p = domain.update_boundary_conditions(diff_p)
            p_eq = (
                conv_p
                # - 1 / self.Mach**2 * (domain.diff(u[0], 0) + domain.diff(u[2], 2)) # u[1]_y is zero
                - 1 / self.Mach**2 * domain.divergence(u)
                + diff_p
            )
            ks.append(jnp.concatenate([u_eq, jnp.array([p_eq])], axis=0))

        for i in range(n_substeps):
            U += dt * b[i] * ks[i]
        new_u = U[:3]
        new_p = U[3]
        new_u_bc = jnp.array(
            [domain.update_boundary_conditions(new_u[i]) for i in range(3)]
        )
        new_p_bc = new_p.at[0, 0, 0].set(0.0)
        U = jnp.concatenate([new_u_bc, jnp.array([new_p_bc])], axis=0)
        return U

    def perform_time_step(
        self,
        U: Optional["jnp_array"] = None,
        __: Optional[Any] = None,
        i: Optional[int] = None,
    ) -> "jnp_array":
        assert U is not None
        assert i is not None
        # return self.perform_explicit_euler_step(U, i)
        return self.perform_rk_step(U, i)

    @partial(jax.jit, static_argnums=(0))
    def solve_scan(self) -> Tuple[Union["jnp_array", VectorField[PhysicalField]], int]:
        cfl_initial = self.get_cfl()
        print_verb("initial cfl:", cfl_initial, debug=True)

        def inner_step_fn(
            u0: Tuple["jnp_array", int], _: Any
        ) -> Tuple[Tuple["jnp_array", int], None]:
            u0_, time_step = u0
            out = self.perform_time_step(u0_, time_step)
            return ((out, time_step + 1), None)

        def step_fn(
            u0: Tuple["jnp_array", int], _: Any
        ) -> Tuple[Tuple["jnp_array", int], Tuple["jnp_array", int]]:
            out, _ = jax.lax.scan(
                jax.checkpoint(inner_step_fn),  # type: ignore[attr-defined]
                # inner_step_fn,
                u0,
                xs=None,
                length=number_of_inner_steps,
                # inner_step_fn, u0, xs=None, length=number_of_inner_steps
            )
            return out, out

        def median_factor(n: int) -> int:
            """Return the median integer factor of n."""
            factors = reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
            factors.sort()
            number_of_factors = len(factors)  # should always be divisible by 2
            return factors[number_of_factors // 2]

        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec
        from jax.sharding import NamedSharding
        from jax.experimental import mesh_utils

        P = jax.sharding.PartitionSpec
        n = jax.local_device_count()
        devices = mesh_utils.create_device_mesh((n,))
        mesh = jax.sharding.Mesh(devices, ("x",))
        sharding = jax.sharding.NamedSharding(mesh, P("x"))  # type: ignore[no-untyped-call]
        u0 = jax.device_put(
            jnp.concatenate(
                [
                    self.get_latest_field("velocity").get_data(),
                    jnp.array([self.get_latest_field("pressure").get_data()]),
                ]
            ),
            sharding,
        )
        ts = jnp.arange(0, self.end_time, self.get_dt())
        number_of_time_steps = len(ts)

        number_of_inner_steps = median_factor(number_of_time_steps)
        # number_of_outer_steps = min(self.max_number_of_outer_steps, number_of_time_steps // number_of_inner_steps)
        number_of_outer_steps = number_of_time_steps // number_of_inner_steps
        # number_of_inner_steps = number_of_time_steps // number_of_outer_steps # TODO

        if self.write_intermediate_output and not self.write_entire_output:
            u_final, trajectory = jax.lax.scan(
                step_fn, (u0, 0), xs=None, length=number_of_outer_steps
            )
            for u in trajectory[0]:
                # u_, _ = u
                velocity = VectorField(
                    [
                        PhysicalField(
                            self.get_physical_domain(),
                            u[i],
                            name="velocity_" + "xyz"[i],
                        )
                        for i in self.all_dimensions()
                    ]
                )
                self.append_field("velocity", velocity, in_place=False)
                pressure = PhysicalField(
                    self.get_physical_domain(), u[3], name="pressure"
                )
                self.append_field("pressure", pressure, in_place=False)
            for i in range(self.get_number_of_fields("velocity")):
                cfl_s = self.get_cfl(i)
                print_verb("i: ", i, "cfl:", cfl_s)
            return (velocity, len(ts))
        elif self.write_entire_output:
            u_final, trajectory = jax.lax.scan(
                step_fn, (u0, 0), xs=None, length=number_of_outer_steps
            )
            velocity_final = VectorField(
                [
                    PhysicalField(
                        self.get_physical_domain(),
                        u_final[0][i],
                        name="velocity_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            self.append_field("velocity", velocity_final, in_place=False)
            pressure_final = PhysicalField(
                self.get_physical_domain(), u_final[0][3], name="pressure"
            )
            self.append_field("pressure", pressure_final, in_place=False)
            cfl_final = self.get_cfl()
            print_verb("final cfl:", cfl_final, debug=True)
            return (trajectory[0], len(ts))
        else:
            u_final, _ = jax.lax.scan(
                step_fn, (u0, 0), xs=None, length=number_of_outer_steps
            )
            velocity_final = VectorField(
                [
                    PhysicalField(
                        self.get_physical_domain(),
                        u_final[0][i],
                        name="velocity_" + "xyz"[i],
                    )
                    for i in self.all_dimensions()
                ]
            )
            self.append_field("velocity", velocity_final, in_place=False)
            cfl_final = self.get_cfl()
            print_verb("final cfl:", cfl_final, debug=True)
            return (velocity_final, len(ts))

    def post_process(self: E) -> None:
        if type(self.post_process_fn) != NoneType:
            assert self.post_process_fn is not None
            for i in range(self.get_number_of_fields("velocity")):
                self.post_process_fn(self, i)  # type: ignore[arg-type]

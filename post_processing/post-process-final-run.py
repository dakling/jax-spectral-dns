#!/usr/bin/env python3

import os
import sys

from matplotlib.pyplot import tight_layout

from jax_spectral_dns.equation import Equation
from jax_spectral_dns.main import get_args_from_yaml_file

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import socket
import yaml
from enum import Enum
from typing import Any, Tuple, Optional, List, Self, cast, Callable
import pyvista as pv
import numpy as np
import h5py
import matplotlib
from matplotlib import figure
from matplotlib.axes import Axes
import jax.numpy as jnp
from jax_spectral_dns.cheb import cheby
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import Field, VectorField, PhysicalField, FourierField
from jax_spectral_dns.navier_stokes_perturbation import (
    NavierStokesVelVortPerturbation,
    update_nonlinear_terms_high_performance_perturbation_skew_symmetric,
)


matplotlib.set_loglevel("error")

from PIL import Image

# matplotlib.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']
font_size = 28
matplotlib.rcParams.update(
    {
        "font.size": font_size,
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)
pv.global_theme.font.size = font_size
pv.global_theme.font.title_size = font_size
pv.global_theme.font.label_size = font_size


# STORE_PREFIX = "/store/DAMTP/dsk34"
# HOME_PREFIX = "/home/dsk34/jax-optim/run"
STORE_PREFIX = "/home/klingenberg/mnt/swirles_store/"
HOME_PREFIX = "/home/klingenberg/mnt/swirles/jax-optim/run/"
STORE_DIR_BASE = os.path.dirname(os.path.realpath(__file__))
HOME_DIR_BASE = STORE_DIR_BASE.replace(STORE_PREFIX, HOME_PREFIX)

args = get_args_from_yaml_file(HOME_DIR_BASE + "/simulation_settings.yml")


class Case:
    STORE_DIR_BASE = (
        "/store/DAMTP/dsk34/"
        if "cam.ac.uk" in socket.gethostname()
        # else "/home/klingenberg/mnt/maths_store/"
        else "/home/klingenberg/mnt/swirles_store/"
    )
    HOME_DIR_BASE = (
        "/home/dsk34/jax-optim/run/"
        if "cam.ac.uk" in socket.gethostname()
        # else "/home/klingenberg/mnt/maths/jax-optim/run/"
        else "/home/klingenberg/mnt/swirles/jax-optim/run/"
    )
    Vel_0_types = Enum(
        "vel_0_types", ["quasilinear", "nonlinear_global", "nonlinear_localised"]
    )

    def __init__(self, directory: str):
        self.directory = directory
        self.T = self.get_T()
        self.e_0 = self.get_e0()
        self.gain = self.get_gain()
        self.Re_tau = self.get_Re_tau()
        self.vel_0 = None
        self.vel_base = None
        self.vel_base_diff_y = None
        self.vel_base_max = None
        self.vel_base_cl = None
        self.vel_base_wall = None
        self.inf_norm = None
        self.dominant_lambda_z = None
        self.dominant_streak_amplitude = None
        self.successfully_read = (
            self.T is not None and self.e_0 is not None and self.gain is not None
        )
        self.vel_0_type = None

    def __repr__(self) -> str:
        return (
            "T: " + str(self.T) + ", e_0: " + str(self.e_0) + ", dir: " + self.directory
        )

    def get_property_from_settings(
        self, property: str, default: Any = None
    ) -> Optional[float]:
        base_path = self.HOME_DIR_BASE
        fname = base_path + "/" + self.directory + "/simulation_settings.yml"
        try:
            with open(fname, "r") as file:
                args = yaml.safe_load(file)
            return args[property]
        except Exception as e:
            return default

    def get_e0(self) -> Optional[float]:
        return self.get_property_from_settings("e_0")

    def get_T(self) -> Optional[float]:
        return self.get_property_from_settings("end_time")

    def get_Re_tau(self) -> Optional[float]:
        return self.get_property_from_settings("Re_tau")

    def get_vel_field_minimal_channel(self):

        domain = self.get_domain()

        cheb_coeffs = np.loadtxt(
            self.HOME_DIR_BASE
            + "/"
            + self.directory
            + "/profiles/Re_tau_180_90_small_channel.csv",
            dtype=np.float64,
        )

        Ny = domain.number_of_cells(1)
        U_mat = np.zeros((Ny, len(cheb_coeffs)))
        for i in range(Ny):
            for j in range(len(cheb_coeffs)):
                U_mat[i, j] = cheby(j, 0)(domain.grid[1][i])
        U_y_slice = U_mat @ cheb_coeffs
        nx, nz = domain.number_of_cells(0), domain.number_of_cells(2)
        u_data = np.moveaxis(
            np.tile(np.tile(U_y_slice, reps=(nz, 1)), reps=(nx, 1, 1)), 1, 2
        )
        vel_base = VectorField(
            [
                PhysicalField(domain, jnp.asarray(u_data)),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
                PhysicalField.FromFunc(domain, lambda X: 0 * X[2]),
            ]
        )
        vel_base.set_name("velocity_base")
        return vel_base

    def get_base_velocity_minimal_channel_slice(self) -> "PhysicalField":
        domain = self.get_domain()
        slice_domain = PhysicalDomain.create(
            (domain.get_shape_aliasing()[1],),
            (False,),
            scale_factors=(1.0,),
            aliasing=1,
        )
        vel_base_turb = self.get_vel_field_minimal_channel()
        vel_base_turb_slice = PhysicalField(slice_domain, vel_base_turb[0][0, :, 0])
        vel_base_turb_slice.set_name("vel_base_minimal_channel")
        return vel_base_turb_slice

    def get_gain(self) -> Optional[float]:
        base_path = self.STORE_DIR_BASE
        phase_space_data_name = (
            base_path + "/" + self.directory + "/plots/phase_space_data.txt"
        )
        try:
            phase_space_data = np.atleast_2d(
                np.genfromtxt(
                    phase_space_data_name,
                    delimiter=",",
                )
            ).T
            return max(phase_space_data[1])
        except Exception:
            return None

    def get_base_velocity(self) -> "PhysicalField":
        if self.vel_base is None:
            base_path = self.STORE_DIR_BASE
            domain = self.get_domain()
            slice_domain = PhysicalDomain.create(
                (domain.get_shape_aliasing()[1],),
                (False,),
                scale_factors=(1.0,),
                aliasing=1,
            )
            try:
                hist_bin = self.directory.split("/")[-1]
                base_velocity_file_path = (
                    base_path
                    + "/"
                    + self.directory
                    + "/fields/vel_hist_bin_"
                    + hist_bin
                )
                vel_base_turb_slice = PhysicalField.FromFile(
                    slice_domain,
                    base_velocity_file_path,
                    "hist_bin_" + hist_bin + "_x",
                    time_step=0,
                )
                self.vel_base = vel_base_turb_slice
            except Exception as e:
                try:
                    base_velocity_file_name = glob(
                        "vel_00_*",
                        root_dir=base_path + "/" + self.directory + "/fields/",
                    )[0]

                    base_velocity_file_path = (
                        base_path
                        + "/"
                        + self.directory
                        + "/fields/"
                        + base_velocity_file_name
                    )
                    vel_base_turb_slice = VectorField.FromFile(
                        slice_domain,
                        base_velocity_file_path,
                        "velocity_spatial_average",
                    )[0]
                    self.vel_base = vel_base_turb_slice
                except Exception as e:
                    raise e
        return self.vel_base

    def get_base_velocity_diff_y(self) -> "PhysicalField":
        if self.vel_base_diff_y is None:
            u_base = self.get_base_velocity()
            self.vel_base_diff_y = u_base.diff(0)
        return self.vel_base_diff_y

    def get_base_velocity_max(self) -> "float":
        if self.vel_base_max is None:
            self.vel_base_max = self.get_base_velocity().max()
        return self.vel_base_max

    def get_base_velocity_at_y(
        self, y: "float"
    ) -> "float":  # TODO take into account on which side the perturbation is located
        grd = self.get_domain().grid
        n_y = jnp.argmin((grd[1] - y)[:-1] * (jnp.roll(grd[1], -1) - y)[:-1])
        return self.get_base_velocity()[n_y]

    def get_base_velocity_at_pm_y(
        self, y: "float"
    ) -> "float":  # TODO take into account on which side the perturbation is located
        grd = self.get_domain().grid
        u_0 = self.get_vel_0()
        u_0_shape = u_0[0].get_data().shape
        max_inds = np.unravel_index(u_0[0].get_data().argmax(axis=None), u_0_shape)
        if max_inds[1] > u_0_shape[1] // 2:
            y = -y
        n_y = jnp.argmin((grd[1] - y)[:-1] * (jnp.roll(grd[1], -1) - y)[:-1])
        return self.get_base_velocity()[n_y]

    def get_base_velocity_cl(self) -> "float":
        if self.vel_base_cl is None:
            self.vel_base_cl = self.get_base_velocity_at_y(0)
        return self.vel_base_cl

    def get_base_velocity_wall(
        self,
    ) -> "float":  # TODO take into account on which side the perturbation is located
        if self.vel_base_wall is None:
            Re_tau = self.Re_tau
            y_wall = 1 - 10.0 / Re_tau
            self.vel_base_wall = self.get_base_velocity_at_pm_y(y_wall)
        return self.vel_base_wall

    def get_base_velocity_diff_y_at_y(self, y: "float") -> "float":
        grd = self.get_domain().grid
        n_y = jnp.argmin((grd[1] - y)[:-1] * (jnp.roll(grd[1], -1) - y)[:-1])
        return self.get_base_velocity_diff_y()[n_y]

    def get_base_velocity_diff_y_at_pm_y(self, y: "float") -> "float":
        grd = self.get_domain().grid
        u_0 = self.get_vel_0()
        u_0_shape = u_0[0].get_data().shape
        max_inds = np.unravel_index(u_0[0].get_data().argmax(axis=None), u_0_shape)
        out_factor = 1.0
        if max_inds[1] > u_0_shape[1] // 2:
            y = -y
            out_factor = -1.0
        n_y = jnp.argmin((grd[1] - y)[:-1] * (jnp.roll(grd[1], -1) - y)[:-1])
        return out_factor * self.get_base_velocity_diff_y()[n_y]

    @classmethod
    def sort_by_e_0(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.e_0)
        return cases

    @classmethod
    def sort_by_u_base_max(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.get_base_velocity_max())
        return cases

    @classmethod
    def sort_by_u_base_cl(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.get_base_velocity_cl())
        return cases

    @classmethod
    def sort_by_u_wall(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.get_base_velocity_wall())
        return cases

    @classmethod
    def sort_by_dir_name(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.directory)
        return cases

    def get_lambdas_over_t(self) -> Optional[Tuple["np_float_array", "np_float_array"]]:
        lambda_file = self.STORE_DIR_BASE + self.directory + "/plots/lambdas.txt"
        lambdas = np.loadtxt(lambda_file).T
        ts = lambdas[0, :]
        lambda_z = lambdas[2, :]
        return (ts, lambda_z)

    def get_domain(self) -> "PhysicalDomain":
        Nx = cast(int, self.get_property_from_settings("Nx", 48))
        Ny = cast(int, self.get_property_from_settings("Ny", 129))
        Nz = cast(int, self.get_property_from_settings("Nz", 80))
        sc_x = cast(float, self.get_property_from_settings("Lx_over_pi", 2.0))
        sc_z = cast(float, self.get_property_from_settings("Lz_over_pi", 1.0))
        domain = PhysicalDomain.create(
            (Nx, Ny, Nz), (True, False, True), (sc_x * np.pi, 1.0, sc_z * np.pi)
        )
        return domain

    def get_vel_0(self) -> "VectorField[PhysicalField]":
        if type(self.vel_0) is NoneType:
            path = self.STORE_DIR_BASE + "/" + self.directory + "/" + "fields/"
            file = "velocity_latest"
            domain = self.get_domain()
            success = False
            retries = 0
            vel_0 = None
            while not success:
                try:
                    vel_0 = VectorField.FromFile(
                        domain, path + file, name="vel_0", allow_projection=True
                    )
                    success = True
                except Exception:
                    try:
                        vel_0 = VectorField.FromFile(
                            domain,
                            path + file,
                            name="velocity_pert",
                            allow_projection=True,
                        )
                        success = True
                    except Exception:
                        try:
                            bak_file = glob("velocity_latest_bak_*", root_dir=path)[0]
                            vel_0 = VectorField.FromFile(
                                domain,
                                path + bak_file,
                                name="vel_0",
                                allow_projection=True,
                            )
                            success = True
                        except Exception:
                            try:
                                bak_file = glob("velocity_latest_bak_*", root_dir=path)[
                                    0
                                ]
                                vel_0 = VectorField.FromFile(
                                    domain,
                                    path + bak_file,
                                    name="velocity_pert",
                                    allow_projection=True,
                                )
                                success = True
                            except Exception as e:
                                if retries < 10:
                                    print_verb(
                                        "issues with opening velocity file for case",
                                        self,
                                        "; trying again in 20 seconds.",
                                    )
                                    time.sleep(20)
                                    retries += 1
                                else:
                                    raise e
            assert vel_0 is not None
            self.vel_0 = vel_0
        return self.vel_0

    def get_velocity_snapshot(
        self, target_time: float
    ) -> Optional["VectorField[FourierField]"]:
        base_path = self.STORE_DIR_BASE
        velocity_trajectory_file_name = (
            base_path + "/" + self.directory + "/fields/trajectory"
        )

        retries = 0
        while retries < 10:
            try:
                with h5py.File(velocity_trajectory_file_name, "r") as f:
                    velocity_trajectory = f["trajectory"]
                    domain = self.get_domain()
                    n_steps = velocity_trajectory.shape[0]
                    end_time = self.T
                    index = int(n_steps * target_time / end_time)
                    vel_hat = VectorField.FromData(
                        FourierField,
                        domain,
                        velocity_trajectory[index],
                        name="velocity_hat",
                        allow_projection=True,
                    )
                    vel_hat.set_time_step(index)
                    return vel_hat
            except Exception:
                print_verb("unable to open trajectory file for", self.directory)
                retries += 1
                time.sleep(20)

    def get_inf_norm(self) -> "float":
        if self.inf_norm is None:
            vel_field = self.get_vel_0()
            # self.inf_norm = max(abs(vel_field[0].get_data().flatten()))
            self.inf_norm = vel_field[0].inf_norm()
        return self.inf_norm

    def get_inf_norm_over_local_base(self) -> "float":
        if self.inf_norm is None:
            self.inf_norm = self.get_inf_norm()
        u_base = self.get_vel_field_minimal_channel()
        u_0 = self.get_vel_0()

        u_0_shape = u_0[0].get_data().shape
        max_inds = np.unravel_index(u_0[0].get_data().argmax(axis=None), u_0_shape)
        u_base_max_loc = u_base[0][*max_inds]
        return self.inf_norm / u_base_max_loc

    def get_dominant_lambda_z(self) -> "float":
        if self.dominant_lambda_z is None:
            target_time = 0.3
            vel_hat = self.get_velocity_snapshot(target_time)
            if vel_hat is not None:
                _, lambda_z = vel_hat[0].get_streak_scales()
                self.dominant_lambda_z = lambda_z * self.Re_tau
            else:
                self.dominant_lambda_z = np.nan
        return self.dominant_lambda_z

    def get_dominant_streak_amplitude(self) -> "float":
        if self.dominant_streak_amplitude is None:
            # target_time = -0.2 # should take from the end
            target_time = 0.8 * self.T
            vel_hat = self.get_velocity_snapshot(target_time)
            if vel_hat is not None:
                self.dominant_streak_amplitude = max(
                    abs(vel_hat[0].field_2d(0).no_hat().get_data().flatten())
                )
            else:
                self.dominant_streak_amplitude = np.nan
        return self.dominant_streak_amplitude

    def classify_vel_0(self) -> "Vel_0_types":
        if self.vel_0_type is not None:
            return self.vel_0_type
        else:
            vel_0_hat = self.get_vel_0().hat()
            vel_0_energy = vel_0_hat.no_hat().energy()
            amplitudes_2d_kx = []
            domain = vel_0_hat[0].get_domain()
            Nx, Ny, Nz = domain.get_shape()
            for kx in range((Nx - 1) // 2 + 1):
                vel_2d_kx = vel_0_hat[0].field_2d(0, kx).no_hat()
                amplitudes_2d_kx.append(vel_2d_kx.max() - vel_2d_kx.min())
            amplitudes_2d_kz = []
            for kz in range((Nz - 1) // 2 + 1):
                vel_2d_kz = vel_0_hat[0].field_2d(2, kz).no_hat()
                amplitudes_2d_kz.append(vel_2d_kz.max() - vel_2d_kz.min())
            kx_max = cast(int, np.argmax(amplitudes_2d_kx))
            kz_max = cast(int, np.argmax(amplitudes_2d_kz))
            vel_0_hat_2d = vel_0_hat.field_2d(0, kx_max).field_2d(2, kz_max)
            if Equation.verbosity_level >= 3:
                coarse_domain = PhysicalDomain.create(
                    (Nx // 4, Ny, Nz // 4),
                    domain.periodic_directions,
                    domain.scale_factors,
                )
                vel_0_filtered_hat = vel_0_hat.filter(coarse_domain)
                print_verb("total energy:", vel_0_energy, verbosity_level=3)
                print_verb("kx, kz:", kx_max, kz_max, verbosity_level=3)
                print_verb(
                    "energy (linear test):",
                    vel_0_hat_2d.no_hat().energy(),
                    verbosity_level=3,
                )
                print_verb(
                    "linear test:",
                    abs((vel_0_hat_2d.no_hat().energy() - vel_0_energy) / vel_0_energy),
                    verbosity_level=3,
                )
                print_verb(
                    "energy (nonlinear test):",
                    vel_0_filtered_hat.no_hat().energy(),
                    verbosity_level=3,
                )
                print_verb(
                    "nonlinear test:",
                    abs(
                        (vel_0_filtered_hat.no_hat().energy() - vel_0_energy)
                        / vel_0_energy
                    ),
                    verbosity_level=3,
                )
            if (
                abs((vel_0_hat_2d.no_hat().energy() - vel_0_energy) / vel_0_energy)
                < 1e-1
            ):
                self.vel_0_type = Case.Vel_0_types["quasilinear"]
                return self.vel_0_type
            else:
                coarse_domain = PhysicalDomain.create(
                    (Nx // 4, Ny, Nz // 4),
                    domain.periodic_directions,
                    domain.scale_factors,
                )
                vel_0_filtered_hat = vel_0_hat.filter(coarse_domain)
                if (
                    abs(
                        (vel_0_filtered_hat.no_hat().energy() - vel_0_energy)
                        / vel_0_energy
                    )
                    < 1e-5
                ):
                    self.vel_0_type = Case.Vel_0_types["nonlinear_global"]
                    return self.vel_0_type
                else:
                    self.vel_0_type = Case.Vel_0_types["nonlinear_localised"]
                    return self.vel_0_type

    def get_marker(self) -> str:
        vel_0_type = self.classify_vel_0()
        if vel_0_type is Case.Vel_0_types["quasilinear"]:
            print_verb("lin", verbosity_level=3)
            return "x"
        elif vel_0_type is Case.Vel_0_types["nonlinear_global"]:
            print_verb("nl glob", verbosity_level=3)
            return "s"
        elif vel_0_type is Case.Vel_0_types["nonlinear_localised"]:
            print_verb("nl loc", verbosity_level=3)
            return "o"
        else:
            print(self)
            print(self.vel_0_type)
            raise Exception("unknown velocity classification.")

    def get_color(self) -> Tuple[str, Any]:
        e_0_min = 1.0e-6
        e_0_max = 1.0e-3
        alpha_min = 0.0
        alpha_max = 1.0
        if self.e_0 < e_0_min:
            out = alpha_min
        else:
            out = (np.log(self.e_0) - np.log(e_0_min)) / (
                np.log(e_0_max) - np.log(e_0_min)
            ) * (alpha_max - alpha_min) + alpha_min
        cmap = matplotlib.colormaps["magma"]
        rgba = cmap(out)
        return rgba, cmap

    def get_alpha(self) -> float:
        return 1.0
        # e_0_min = 1.0e-6
        # e_0_max = 1.0e-3
        # alpha_min = 0.1
        # alpha_max = 1.0
        # if self.e_0 < e_0_min:
        #     return alpha_min
        # else:
        #     return (np.log(self.e_0) - np.log(e_0_min)) / (
        #         np.log(e_0_max) - np.log(e_0_min)
        #     ) * (alpha_max - alpha_min) + alpha_min

    def get_classification_name(self) -> str:
        vel_0_type = self.classify_vel_0()
        if vel_0_type is Case.Vel_0_types["quasilinear"]:
            print_verb("lin", verbosity_level=3)
            return "quasi-linear regime"
        elif vel_0_type is Case.Vel_0_types["nonlinear_global"]:
            print_verb("nl glob", verbosity_level=3)
            return "nonlinear global regime"
        elif vel_0_type is Case.Vel_0_types["nonlinear_localised"]:
            print_verb("nl loc", verbosity_level=3)
            return "nonlinear localised regime"
        else:
            print(self)
            print(self.vel_0_type)
            raise Exception("unknown velocity classification.")

    def append_to(
        self, cases: "List[Self]", prune_duplicates: bool = True
    ) -> "List[Self]":
        if self.successfully_read:
            assert self.gain is not None
            Ts = [case.T for case in cases]
            e_0s = [case.e_0 for case in cases]
            gains = [case.gain for case in cases]
            if prune_duplicates and (self.T in Ts) and (self.e_0 in e_0s):
                ind = e_0s.index(self.e_0)
                other_gain = gains[ind]
                assert other_gain is not None
                if self.gain > other_gain:
                    cases[ind] = self
            else:
                cases.append(self)
        return cases

    @classmethod
    def prune_times(cls, cases: "List[Case]") -> "List[Case]":
        Ts = [case.T for case in cases]
        dominant_time = max(set(Ts), key=Ts.count)
        return [case for case in cases if abs(case.T - dominant_time) < 1.0e-10]

    @classmethod
    def collect(
        cls,
        base_path: str,
        prune_duplicates: bool = True,
        with_linear: bool = True,
        legal_names: List[str] = [
            "[0-9]eminus[0-9]",
            "[0-9]eminus[0-9]_sweep_*",
            "[0-9]eminus[0-9]_from_*",
        ],
    ) -> "List[Self]":
        home_path = Case.HOME_DIR_BASE + "/" + base_path
        dirs = glob(legal_names[0], root_dir=home_path)
        for legal_name in legal_names[1:]:
            dirs += glob(legal_name, root_dir=home_path)
        cases = []
        for dir in dirs:
            case = cls(base_path + "/" + dir)
            cases = case.append_to(cases, prune_duplicates)
        if with_linear:
            case = cls(base_path + "/" + "linear")
            case.e_0 = 0.0
            cases = case.append_to(cases)
        return cases

    @classmethod
    def get_e_0_lam_lower_boundary(cls, cases: "List[Case]") -> "float":
        return [0.0 for _ in cases]

    @classmethod
    def get_e_0_lam_upper_boundary(cls, cases: "List[Case]") -> "float":
        cases_sorted = Case.sort_by_e_0(cases)
        for i in range(len(cases_sorted) - 1):
            if (
                cases[i].classify_vel_0() == Case.Vel_0_types["quasilinear"]
                and cases[i + 1].classify_vel_0() != Case.Vel_0_types["quasilinear"]
            ):
                return cast(float, cases[i].e_0)
        return cast(float, cases[-1].e_0)

    @classmethod
    def get_e_0_nl_glob_lower_boundary(cls, cases: "List[Case]") -> "Optional[float]":
        cases_sorted = Case.sort_by_e_0(cases)
        for i in range(1, len(cases_sorted)):
            if (
                cases[i].classify_vel_0() == Case.Vel_0_types["nonlinear_global"]
                and cases[i - 1].classify_vel_0()
                != Case.Vel_0_types["nonlinear_global"]
            ):
                return cases[i].e_0
        return None

    @classmethod
    def get_e_0_nl_glob_upper_boundary(cls, cases: "List[Case]") -> "Optional[float]":
        cases_sorted = Case.sort_by_e_0(cases)
        for i in range(len(cases_sorted) - 1):
            if (
                cases[i].classify_vel_0() == Case.Vel_0_types["nonlinear_global"]
                and cases[i + 1].classify_vel_0()
                != Case.Vel_0_types["nonlinear_global"]
            ):
                return cases[i].e_0
        if cases[-1].classify_vel_0() == Case.Vel_0_types["nonlinear_global"]:
            return cases[-1].e_0
        return None

    @classmethod
    def get_e_0_nl_loc_lower_boundary(cls, cases: "List[Case]") -> "Optional[float]":
        cases_sorted = Case.sort_by_e_0(cases)
        for i in range(1, len(cases_sorted)):
            if (
                cases[i].classify_vel_0() == Case.Vel_0_types["nonlinear_localised"]
                and cases[i - 1].classify_vel_0()
                != Case.Vel_0_types["nonlinear_localised"]
            ):
                return cases[i].e_0
        return None

    @classmethod
    def get_e_0_nl_loc_upper_boundary(cls, cases: "List[Case]") -> "Optional[float]":
        cases_sorted = Case.sort_by_e_0(cases)
        for i in range(len(cases_sorted) - 1):
            if (
                cases[i].classify_vel_0() == Case.Vel_0_types["nonlinear_localised"]
                and cases[i + 1].classify_vel_0()
                != Case.Vel_0_types["nonlinear_localised"]
            ):
                return cases[i].e_0
        if cases[-1].classify_vel_0() == Case.Vel_0_types["nonlinear_localised"]:
            return cases[-1].e_0
        return None


def collect(dirs_and_names: Any) -> List["Case"]:
    all_cases = []
    for base_path, _, _ in dirs_and_names:
        print_verb("collecting cases in", base_path)
        cases = Case.collect(base_path, with_linear=False)
        cases = Case.prune_times(cases)  # TODO why is this necessary?
        all_cases += cases
        print_verb(
            "collected cases:",
            "; ".join([case.directory.split("/")[-1] for case in cases]),
        )
    return all_cases


class CessCase(Case):
    def __init__(self, directory: str, A=None, K=None):
        super().__init__(directory)
        if A is None:
            self.A = self.get_property_from_settings("cess_mean_a", 53.516)
        else:
            self.A = A
        if K is None:
            self.K = self.get_property_from_settings("cess_mean_k", 0.677)
        else:
            self.K = K

    def get_base_velocity(self) -> "PhysicalField":
        if self.vel_base is None:
            Re_tau = self.get_Re_tau()
            assert Re_tau is not None

            def get_vel_field_cess(
                A: float = 25.4,
                K: float = 0.426,
            ) -> PhysicalField:

                u_sc: float = 1.0
                domain = self.get_domain()
                slice_domain = PhysicalDomain.create(
                    (domain.get_shape_aliasing()[1],),
                    (False,),
                    scale_factors=(1.0,),
                    aliasing=1,
                )

                def nu_t_fn(X: "np_jnp_array") -> "jsd_float":
                    def get_y(y_in: "float") -> "float":
                        return min((1 - y_in), (1 + y_in))

                    y = np.vectorize(get_y)(X[0])
                    B = 1.0  # pressure gradient
                    Re = Re_tau
                    return (
                        1
                        / 2
                        * (
                            (
                                1
                                + K**2
                                * Re**2
                                * B**2
                                / 9
                                * (2 * y - y**2) ** 2
                                * (3 - 4 * y + 2 * y**2) ** 2
                                * (1 - np.exp(-y * Re * B**0.5 / A)) ** 2
                            )
                            ** 0.5
                        )
                        + 0.5
                    )

                def shear_fn(X: "np_jnp_array") -> "jsd_float":
                    def get_y(y_in: "float") -> Tuple["float", "float"]:
                        return min((1 - y_in), (1 + y_in)), np.sign(y_in)

                    y, sign = np.vectorize(get_y)(X[0])
                    return -sign * Re_tau * (1 - y) / nu_t_fn(X)

                shear = PhysicalField.FromFunc(
                    slice_domain, cast("Vel_fn_type", shear_fn), name="shear"
                )
                vel_base = shear.integrate(0, bc_left=0.0) * u_sc
                vel_base.set_name("velocity_base")
                return vel_base

            vel_base_turb = get_vel_field_cess(self.A, self.K)
            self.vel_base = vel_base_turb
        return self.vel_base


class PertCase(Case):
    def __init__(self, directory: str, mean_perturbation_=None):
        super().__init__(directory)
        if mean_perturbation_ is None:
            self.pert = self.get_property_from_settings("mean_perturbation", 0.0)
        else:
            self.pert = mean_perturbation_

    def get_base_velocity(self) -> "PhysicalField":

        if self.vel_base is None:
            domain = self.get_domain()
            slice_domain = PhysicalDomain.create(
                (domain.get_shape_aliasing()[1],),
                (False,),
                scale_factors=(1.0,),
                aliasing=1,
            )
            vel_base_perturbation = PhysicalField.FromFunc(
                slice_domain,
                lambda X: self.pert
                * (jnp.cos(jnp.pi * X[0]) + jnp.cos(2 * jnp.pi * X[0])),
            )

            vel_base_turb_slice = self.get_base_velocity_minimal_channel_slice()
            vel_base = (
                vel_base_turb_slice + vel_base_perturbation
            )  # continuously blend from turbulent to laminar mean profile

            self.vel_base = vel_base
        return self.vel_base


def post_process(
    file: str,
    end_time: float,
    Lx_over_pi: float,
    Lz_over_pi: float,
    Re_tau: float,
    time_step_0: int = 0,
) -> None:

    dir = HOME_DIR_BASE.replace(HOME_PREFIX, "")
    if args.get("mean_perturbation") is not None:
        case = PertCase(dir)
    elif args.get("cess_mean_a") is not None or args.get("cess_mean_k") is not None:
        case = CessCase(dir)
    else:
        case = Case(dir)

    domain = case.get_domain()
    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        n_steps = velocity_trajectory.shape[0]
        # fourier_domain = domain.hat()

        ts = []
        energy_t = []
        energy_x_2d = []
        energy_x_3d = []
        energy_x_2d_1 = []
        energy_x_2d_2 = []
        amplitude_t = []
        amplitude_x_2d_t = []
        amplitude_x_2d_t_1 = []
        amplitude_x_2d_t_2 = []
        amplitude_z_2d_t = []
        amplitude_z_2d_t_1 = []
        amplitude_z_2d_t_2 = []
        amplitude_3d_t = []
        amplitudes_2d_kxs = []
        amplitudes_2d_kzs = []
        amplitudes_2d_vilda = []
        lambda_y_s = []
        lambda_z_s = []
        try:
            E_0 = case.get_vel_field_minimal_channel().energy()
        except FileNotFoundError:
            E_0 = 1.0
        # prod = []
        # diss = []
        print("preparing")
        for j in range(n_steps):
            print("preparing, step", j + 1, "of", n_steps)
            vel_hat_ = VectorField.FromData(
                FourierField, domain, velocity_trajectory[j], name="velocity_hat"
            )
            vel_hat_.set_time_step(j + time_step_0)
            vel_ = vel_hat_.no_hat()
            vel_.set_time_step(j + time_step_0)

            vel_energy_ = vel_.energy()
            time_ = (vel_.get_time_step() / (n_steps - 1)) * end_time
            ts.append(time_)
            energy_t.append(vel_energy_)
            # e_x_2d = vel_[0].hat().energy_2d(0)
            e_x_2d = vel_.hat().energy_2d(0)
            e_x_3d = vel_energy_ - e_x_2d
            energy_x_2d.append(e_x_2d)
            energy_x_3d.append(e_x_3d)
            # amplitude_t.append(vel_[0].max() - vel_[0].min())
            amplitude_t.append(vel_[0].inf_norm())
            vel_2d_x = vel_hat_[0].field_2d(0).no_hat()
            vel_2d_x_1 = vel_hat_[0].field_2d(0, 1).no_hat()
            vel_2d_x_2 = vel_hat_[0].field_2d(0, 2).no_hat()
            vel_2d_z = vel_hat_[0].field_2d(2).no_hat()
            vel_2d_z_1 = vel_hat_[0].field_2d(2, 1).no_hat()
            vel_2d_z_2 = vel_hat_[0].field_2d(2, 2).no_hat()
            e_x_2d_1 = vel_2d_x_1.energy()
            e_x_2d_2 = vel_2d_x_2.energy()
            energy_x_2d_1.append(e_x_2d_1)
            energy_x_2d_2.append(e_x_2d_2)
            vel_2d_x.set_name("velocity_x_2d")
            vel_2d_x.set_time_step(j)
            vel_2d_x.plot_3d(0, rotate=True)
            vel_2d_x.plot_3d(2)
            amplitude_x_2d_t.append(vel_2d_x.inf_norm())
            amplitude_x_2d_t_1.append(vel_2d_x_1.inf_norm())
            amplitude_x_2d_t_2.append(vel_2d_x_2.inf_norm())
            amplitude_z_2d_t.append(vel_2d_z.inf_norm())
            amplitude_z_2d_t_1.append(vel_2d_z_1.inf_norm())
            amplitude_z_2d_t_2.append(vel_2d_z_2.inf_norm())
            Nx = domain.get_shape()[0]
            Nz = domain.get_shape()[2]
            amplitudes_2d_kx = []
            for kx in range((Nx - 1) // 2 + 1):
                vel_2d_kx = vel_hat_[0].field_2d(0, kx).no_hat()
                # amplitudes_2d_kx.append(vel_2d_kx.max() - vel_2d_kx.min())
                amplitudes_2d_kx.append(vel_2d_kx.inf_norm())
            amplitudes_2d_kz = []
            for kz in range((Nz - 1) // 2 + 1):
                vel_2d_kz = vel_hat_[0].field_2d(2, kz).no_hat()
                # amplitudes_2d_kz.append(vel_2d_kz.max() - vel_2d_kz.min())
                amplitudes_2d_kz.append(vel_2d_kz.inf_norm())
            amplitudes_2d_kxs.append(amplitudes_2d_kx)
            amplitudes_2d_kzs.append(amplitudes_2d_kz)

            # kx_max = np.argmax(amplitudes_2d_kx)
            # kz_max = np.argmax(amplitudes_2d_kz)
            # vel_0_hat_2d = vel_hat_.field_2d(0, kx_max).field_2d(2, kz_max)
            # energy_of_highest_mode_ratio = abs(
            #     (vel_0_hat_2d.no_hat().energy() - vel_energy_) / vel_energy_
            # )
            # print("energy_of_highest_mode_ratio:", energy_of_highest_mode_ratio)

            energy_arr = np.vstack([np.array(ts), np.array(energy_t)])
            np.savetxt("plots/energy.txt", energy_arr.T)

            # fig = figure.Figure()
            # ax = fig.subplots(2, 1)
            # ax[0].plot(amplitudes_2d_kx, "k.")
            # ax[1].plot(amplitudes_2d_kz, "k.")
            # fig.tight_layout()
            # fig.savefig(
            #     "plots/plot_amplitudes_over_wns_t_" + "{:06}".format(j) + ".png",
            #     bbox_inches="tight",
            # )
            # vel_3d = vel_ - VectorField(
            #     [vel_2d_x, PhysicalField.Zeros(domain), PhysicalField.Zeros(domain)]
            # )
            # # amplitude_3d_t.append(vel_3d.max() - vel_3d.min())
            # # amplitude_3d_t.append(vel_3d[0].max() - vel_3d[0].min())
            # amplitude_3d_t.append(vel_3d[0].inf_norm())
            # # prod.append(nse.get_production(j))
            # # diss.append(nse.get_dissipation(j))
            # amplitudes_2d_vilda.append(np.sqrt(vel_2d_x.energy() / E_0 * Re_tau))

            lambda_y, lambda_z = vel_hat_[0].get_streak_scales()
            print("lambda_y+:", lambda_y * Re_tau)
            print("lambda_z+:", lambda_z * Re_tau)
            lambda_y_s.append(lambda_y)
            lambda_z_s.append(lambda_z)

        amplitudes_x_arr = np.vstack(
            [np.atleast_2d(np.array(ts)), np.atleast_2d(np.array(amplitudes_2d_kxs)).T]
        )
        np.savetxt("plots/amplitudes_x.txt", amplitudes_x_arr.T)
        amplitudes_z_arr = np.vstack(
            [np.atleast_2d(np.array(ts)), np.atleast_2d(np.array(amplitudes_2d_kzs)).T]
        )
        np.savetxt("plots/amplitudes_z.txt", amplitudes_z_arr.T)

        # fig = figure.Figure()
        # ax = fig.subplots(1, 1)
        # ax.plot(ts, amplitudes_2d_vilda, "k.")
        # ax.set_xlabel("$t u_\\tau / h$")
        # ax.set_ylabel("$A \\sqrt{\\text{Re}_\\tau} $")
        # fig.tight_layout()
        # fig.savefig(
        #     "plots/plot_amplitudes_vilda.png",
        #     bbox_inches="tight",
        # )
        energy_t_arr = np.array(energy_t)
        # energy_x_2d_arr = np.array(energy_x_2d)
        # energy_x_2d_1_arr = np.array(energy_x_2d_1)
        # energy_x_2d_2_arr = np.array(energy_x_2d_2)
        # energy_x_3d_arr = np.array(energy_x_3d)
        print(max(energy_t_arr) / energy_t_arr[0])

        amplitudes_2d_kxs_arr = np.array(amplitudes_2d_kxs)
        amplitudes_2d_kzs_arr = np.array(amplitudes_2d_kzs)
        # fig_kx_pub = figure.Figure()
        # ax_kx_pub = fig_kx_pub.subplots(1, 1)
        fig_k_size = (12, 9)
        fig_kx = figure.Figure(figsize=fig_k_size)
        ax_kx = fig_kx.subplots(1, 1)
        fig_kz = figure.Figure(figsize=fig_k_size)
        ax_kz = fig_kz.subplots(1, 1)
        ax_kx.set_yscale("log")
        ax_kz.set_yscale("log")
        ax_kx.set_xlabel("$t u_\\tau / h$")
        ax_kz.set_xlabel("$t u_\\tau / h$")
        ax_kx.set_ylabel("$|u|_{\\infty}$")
        ax_kz.set_ylabel("$|u|_{\\infty}$")
        # ax_kx_pub.plot(amplitude_t, "k--", label="full")
        ax_kx.plot(ts, amplitude_t, "k--", label="full")
        ax_kz.plot(ts, amplitude_t, "k--", label="full")
        for kx in range((Nx - 1) // 2 + 1)[:10]:
            kx_ = int(kx * 2 / Lx_over_pi)
            ax_kx.plot(
                ts, amplitudes_2d_kxs_arr[:, kx], label="$k_x = " + str(kx_) + "$"
            )
            # ax_kx_pub.plot(amplitudes_2d_kxs_arr[:, kx], "-" label="kx = " + str(kx_))
        for kz in range((Nz - 1) // 2 + 1)[:10]:
            kz_ = int(kz * 2 / Lz_over_pi)
            ax_kz.plot(
                ts, amplitudes_2d_kzs_arr[:, kz], label="$k_z = " + str(kz_) + "$"
            )
        # fig_kx.legend()
        ax_kx.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_kx.tight_layout()
        fig_kx.savefig("plots/plot_amplitudes_kx" + ".png", bbox_inches="tight")
        # fig_kz.legend()
        ax_kz.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_kz.tight_layout()
        fig_kz.savefig("plots/plot_amplitudes_kz" + ".png", bbox_inches="tight")

        # fig_lambdas = figure.Figure()
        # ax_lambdas = fig_lambdas.subplots(1, 1)
        # ax_lambdas2 = ax_lambdas.twinx()
        # ax_lambdas.plot(ts, Re_tau * np.array(lambda_y_s), "ko", label="$\\lambda_y^+$")
        # ax_lambdas2.plot(
        #     ts, Re_tau * np.array(lambda_z_s), "bo", label="$\\lambda_z^+$"
        # )
        # ax_lambdas.set_xlabel("$t$")
        # ax_lambdas.set_ylabel("$\\lambda_y^+$")
        # ax_lambdas2.set_ylabel("$\\lambda_z^+$", color="blue")
        # ax_lambdas2.tick_params(axis="y", labelcolor="blue")
        # ax_lambdas.set_ylim(bottom=0)
        # ax_lambdas2.set_ylim(bottom=0)
        # # fig_lambdas.legend()
        # fig_lambdas.tight_layout()
        # fig_lambdas.savefig("plots/plot_lambdas" + ".png", bbox_inches="tight")

        # fig_lambda_z = figure.Figure()
        # ax_lambda_z = fig_lambda_z.subplots(1, 1)
        # ax_lambda_z.plot(
        #     ts, Re_tau * np.array(lambda_z_s), "bo", label="$\\lambda_z^+$"
        # )
        # ax_lambda_z.plot(
        #     ts, [100.0 for _ in ts], "k-", label="$\\lambda_z^+_\\text{mean}$"
        # )
        # ax_lambda_z.plot(
        #     ts,
        #     [60.0 for _ in ts],
        #     "k--",
        #     label="$\\lambda_z^+_\\text{mean} - \\lambda_z^+_\\text{std}$",
        # )
        # ax_lambda_z.plot(
        #     ts,
        #     [140.0 for _ in ts],
        #     "k--",
        #     label="$\\lambda_z^+_\\text{mean} + \\lambda_z^+_\\text{std}$",
        # )
        # ax_lambda_z.axvline(x=0.1, linestyle="--", color="k")
        # ax_lambda_z.axvline(x=0.7, linestyle="--", color="k")
        # ax_lambda_z.set_xlabel("$t$")
        # ax_lambda_z.set_ylim(bottom=0)
        # # fig_lambda_z.legend()
        # fig_lambda_z.tight_layout()
        # fig_lambda_z.savefig("plots/plot_lambda_z" + ".png", bbox_inches="tight")

        # lambda_arr = np.vstack(
        #     [np.array(ts), np.array(lambda_y_s), np.array(lambda_z_s)]
        # )
        # np.savetxt("plots/lambdas.txt", lambda_arr.T)

        try:
            vel_base = case.get_base_velocity()
        except FileNotFoundError:
            vel_base = vel_ * 0.0

        print("main post-processing loop")
        for i in range(n_steps):
            print("step", i + 1, "of", n_steps)
            # time = (i / (n_steps - 1)) * end_time
            vel_hat = VectorField.FromData(
                FourierField, domain, velocity_trajectory[i], name="velocity"
            )
            vel = vel_hat.no_hat()
            vel.set_time_step(i + time_step_0)

            # vort = vel.curl()
            vel.set_name("velocity")
            # vort.set_name("vorticity")
            time_step = vel.get_time_step()

            if i == 0:
                vel_shape = vel[0].get_data().shape
                max_inds = np.unravel_index(
                    vel[0].get_data().argmax(axis=None), vel_shape
                )
                Nx, _, Nz = vel_shape
                x_max = max_inds[0] / Nx * domain.grid[0][-1]
                z_max = max_inds[2] / Nz * domain.grid[2][-1]
            # vel_total_x = vel[0] + vel_base[0]
            # vel_total_x.set_name("velocity_total_x")
            # vel_total_x.plot_3d(2, z_max, name="$U_x$")
            # vel[0].plot_3d(2, z_max, name="$\\tilde{u}_x$", name_color="red")
            vel[0].plot_3d(2, z_max, name="$\\tilde{u}_x$", flip_axis=1)
            # vel[1].plot_3d(2, z_max)
            # vel[2].plot_3d(2, z_max)
            vel[0].plot_3d(
                # 0, x_max, rotate=True, name="$\\tilde{u}_x$", name_color="red"
                0,
                x_max,
                rotate=True,
                name="$\\tilde{u}_x$",
                flip_axis=1,
            )
            vel[0].plot_3d(
                0, x_max, rotate=True, name="$\\tilde{u}_x$", no_cb=True, flip_axis=1
            )

            slice_domain = PhysicalDomain.create(
                (domain.get_shape_aliasing()[1],),
                (False,),
                scale_factors=(1.0,),
                aliasing=1,
            )
            vel_0_base_no_slice = vel_hat.field_2d(2).field_2d(0).no_hat()[0]

            vel_0_base = PhysicalField(slice_domain, vel_0_base_no_slice[0, :, 0])
            vel_0_base.set_name("vel_dist_base")
            vel_0_base.set_time_step(time_step)
            vel_0_base.plot_center(1)

            vel_base_inst = vel_0_base + vel_base
            vel_base_inst.set_name("vel_base_inst")
            vel_base_inst.set_time_step(time_step)
            vel_base_inst.plot_center(0, vel_base)
            # vel[1].plot_3d(0, x_max, rotate=True)
            # vel[2].plot_3d(0, x_max, rotate=True)
            # vel.plot_streamlines(2)
            # vel[1].plot_isolines(2)
            # vel[0].plot_isosurfaces(name="$u_x$", name_color="red")
            if i == 0:
                vel[0].plot_isosurfaces(
                    name="$\\tilde{u}_x$", flip_axis=1, add_axes=True
                )
            else:
                vel[0].plot_isosurfaces(
                    # name="$\\tilde{u}_x$", flip_axis=1, add_axes=False
                    name="$\\tilde{u}_x$",
                    flip_axis=1,
                    add_axes=True,
                )
            # vel[1].plot_isosurfaces()
            # vel[2].plot_isosurfaces()

            # # pressure
            # pressure_poisson_source = PhysicalField.Zeros(domain)
            # for k in range(3):
            #     for j in range(3):
            #         pressure_poisson_source += -(vel[k] * vel[j]).diff(j).diff(k)
            #         pressure_poisson_source += -(vel_base[k] * vel[j]).diff(j).diff(k)
            #         pressure_poisson_source += -(vel[k] * vel_base[j]).diff(j).diff(k)
            # pressure_poisson_source.set_name("pressure_poisson_source")
            # pressure_poisson_source.plot_3d(0)
            # pressure_poisson_source.plot_3d(2)
            # pressure = pressure_poisson_source.hat().solve_poisson().no_hat()
            # # filter_field = PhysicalField.FromFunc(domain, lambda X: jnp.exp(1.0)**(- 20.0 * X[1]**10) + 0.0 * X[2])
            # # filter_field = PhysicalField.FromFunc(
            # #     domain, lambda X: jnp.exp(-((1.03 * X[1]) ** 40)) + 0.0 * X[2]
            # # )  # make sure that we are not messing with the boundary conditions
            # # pressure *= filter_field
            # pressure.update_boundary_conditions()
            # pressure.set_name("pressure")
            # pressure.set_time_step(vel.get_time_step())
            # if i == 0:
            #     vel_shape = vel[0].get_data().shape
            #     max_inds = np.unravel_index(
            #         pressure.get_data().argmax(axis=None), vel_shape
            #     )
            #     Nx, _, Nz = vel_shape
            #     x_max_pres = max_inds[0] / Nx * domain.grid[0][-1]
            #     z_max_pres = max_inds[2] / Nz * domain.grid[2][-1]
            # pressure.plot_3d(0, x_max_pres, rotate=True)
            # pressure.plot_3d(2, z_max_pres)
            # dp_dy = pressure.diff(1)
            # dp_dy.update_boundary_conditions()
            # dp_dy.set_name("dp_dy")
            # dp_dy.set_time_step(vel.get_time_step())
            # if i == 0:
            #     vel_shape = vel[0].get_data().shape
            #     max_inds = np.unravel_index(
            #         dp_dy.get_data().argmax(axis=None), vel_shape
            #     )
            #     Nx, _, Nz = vel_shape
            #     x_max_dpdy = max_inds[0] / Nx * domain.grid[0][-1]
            #     z_max_dpdy = max_inds[2] / Nz * domain.grid[2][-1]
            # dp_dy.plot_3d(0, x_max_dpdy, rotate=True)
            # dp_dy.plot_3d(2, z_max_dpdy)

            # q_crit = vel.get_q_criterion()
            # q_crit.set_time_step(vel.get_time_step())
            # q_crit.set_name("q_criterion")
            # q_crit.plot_3d(2, z_max)
            # q_crit.plot_3d(0, x_max)
            # vel.plot_q_criterion_isosurfaces(iso_vals=[0.05, 0.1, 0.5])
            # vel.plot_wavenumbers(1)
            # vel.magnitude().plot_wavenumbers(1)

            fig = figure.Figure()
            ax = fig.subplots(1, 1)
            assert type(ax) is Axes
            ax.set_xlabel("$t u_\\tau / h$")
            ax.set_ylabel("$G$")
            if i == 0:
                fig_amplitudes = figure.Figure()
                ax_amplitudes = fig_amplitudes.subplots(1, 1)
                assert type(ax_amplitudes) is Axes
                # fig_pd = figure.Figure()
                # ax_pd = fig_pd.subplots(1, 1)
                # assert type(ax_pd) is Axes

                # prod_arr = np.array(prod)
                # diss_arr = np.array(diss)
                # ax.plot(ts, energy_t_arr / energy_t_arr[0], "k-")
                # ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
                # ax.plot(
                #     ts[: i + 1],
                #     energy_t_arr[: i + 1] / energy_t_arr[0],
                #     "ko",
                #     label="$G$",
                # )
                # ax.plot(ts, energy_x_2d_arr / energy_t_arr[0], "b.")
                # # ax.plot(ts, energy_x_2d_arr / energy_x_2d_arr[0], "b.")
                # # ax.plot(ts, energy_x_2d_1_arr / energy_t_arr[0], "y.")
                # # ax.plot(ts, energy_x_2d_2_arr / energy_t_arr[0], "m.")
                # ax.plot(ts, energy_x_3d_arr / energy_t_arr[0], "g.")
                # # ax.plot(ts, energy_x_3d_arr / energy_x_3d_arr[0], "g.")
                # ax.plot(
                #     ts[: i + 1],
                #     energy_x_2d_arr[: i + 1] / energy_t_arr[0],
                #     # energy_x_2d_arr[: i + 1] / energy_x_2d_arr[0],
                #     "bo",
                #     label="$G_{k_x = 0}$",
                # )
                # ax.plot(
                #     ts[: i + 1],
                #     energy_x_3d_arr[: i + 1] / energy_t_arr[0],
                #     # energy_x_3d_arr[: i + 1] / energy_x_3d_arr[0],
                #     "go",
                #     label="$G_{3d}$",
                # )
                # ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
                # ax.set_box_aspect(1)
                fig.savefig(
                    Field.plotting_dir + "/plot_energy" + ".png",
                    bbox_inches="tight",
                )
            ax.plot(ts, energy_t_arr / energy_t_arr[0], "k.")
            ax.plot(
                ts[: i + 1],
                energy_t_arr[: i + 1] / energy_t_arr[0],
                "ko",
                label="$G$",
            )
            fig.savefig(
                Field.plotting_dir
                + "/plot_energy_t_"
                + "{:06}".format(time_step)
                + ".png",
                bbox_inches="tight",
            )
            # # ax_2d_over_3d.set_yscale("log")
            # ax_amplitudes.plot(ts, amplitude_t, "k.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_t[: i + 1],
            #     "ko",
            #     label="total perturbation x-velocity amplitude",
            # )
            # ax_amplitudes.plot(ts, amplitude_x_2d_t, "b.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_x_2d_t[: i + 1],
            #     "bo",
            #     label="streak (k_x = 0) amplitude (x-velocity)",
            # )
            # ax_amplitudes.plot(ts, amplitude_x_2d_t_1, "y.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_x_2d_t_1[: i + 1],
            #     "yo",
            #     label="amplitude (kx = 1) (x-velocity)",
            # )
            # ax_amplitudes.plot(ts, amplitude_x_2d_t_2, "r.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_x_2d_t_2[: i + 1],
            #     "ro",
            #     label="amplitude (kx = 2)",
            # )
            # ax_amplitudes.plot(ts, amplitude_z_2d_t, "m.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_z_2d_t[: i + 1],
            #     "mo",
            #     label="kz = 0 amplitude",
            # )
            # ax_amplitudes.plot(ts, amplitude_z_2d_t_1, "c.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_z_2d_t_1[: i + 1],
            #     "co",
            #     label="amplitude (kz = 1)",
            # )
            # ax_amplitudes.plot(ts, amplitude_z_2d_t_2, "k.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_z_2d_t_2[: i + 1],
            #     "ko",
            #     label="amplitude (kz = 2)",
            # )
            # ax_amplitudes.plot(ts, amplitude_3d_t, "g.")
            # ax_amplitudes.plot(
            #     ts[: i + 1],
            #     amplitude_3d_t[: i + 1],
            #     "go",
            #     label="perturbation amplitude w/o streak (x-velocity)",
            # )
            # ax_amplitudes.set_xlabel("$t u_\\tau / h$")
            # ax_amplitudes.set_ylabel("$A$")
            # fig_amplitudes.legend()
            # fig_amplitudes.tight_layout()
            # fig_amplitudes.savefig(
            #     Field.plotting_dir
            #     + "/plot_amplitudes_t_"
            #     + "{:06}".format(time_step)
            #     + ".png",
            #     bbox_inches="tight",
            # )

            # fig_kx = figure.Figure()
            # ax_kx = fig_kx.subplots(1, 1)
            # fig_kz = figure.Figure()
            # ax_kz = fig_kz.subplots(1, 1)
            # ax_kx.set_xlabel("$t u_\\tau / h$")
            # ax_kx.set_ylabel(
            #     # "$\\textcolor{red}{\\tilde{u}_{x_\\text{max}}} - \\textcolor{red}{\\tilde{u}_{x_\\text{min}}}$"
            #     # "$\\tilde{u}_{x_\\text{max}} - \\tilde{u}_{x_\\text{min}}$"
            #     # "$\\tilde{u}_{x_\\text{maax}} - \\tilde{u}_{x_\\text{min}}$"
            #     "$|\\tilde{u}_{x}|_\\text{inf}$",
            # )
            # # ax_kx.set_ylabel("${\\tilde{u}_x}$ amplitude")
            # # ax_kx.yaxis.label.set_color("red")
            # ax_kz.set_xlabel("$t u_\\tau / h$")
            # # ax_kz.set_ylabel("$\\textcolor{red}{\\tilde{u}_x}$ amplitude")
            # ax_kz.set_ylabel(
            #     # "$\\textcolor{red}{\\tilde{u}_{x_\\text{max}}} - \\textcolor{red}{\\tilde{u}_{x_\\text{min}}}$"
            #     # "$\\tilde{u}_{x_\\text{max}} - \\tilde{u}_{x_\\text{min}}$"
            #     "$|\\tilde{u}_{x}|_\\text{inf}$",
            # )
            # # ax_kz.set_ylabel("${\\tilde{u}_x}$ amplitude")
            # # ax_kz.yaxis.label.set_color("red")
            # ax_kx.plot(ts, amplitude_t, "k.")
            # ax_kx.plot(ts[: i + 1], amplitude_t[: i + 1], "ko", label="full")
            # ax_kz.plot(ts, amplitude_t, "k.")
            # ax_kz.plot(ts[: i + 1], amplitude_t[: i + 1], "ko", label="full")
            # # for kx in range((Nx - 1) // 2 + 1)[0:14:2]:
            # for kx in range((Nx - 1) // 2 + 1)[0:10]:
            #     dots = ax_kx.plot(ts, amplitudes_2d_kxs_arr[:, kx], ".")
            #     ax_kx.plot(
            #         ts[: i + 1],
            #         amplitudes_2d_kxs_arr[: i + 1, kx],
            #         "o",
            #         color=dots[0].get_color(),
            #         label="$k_x = " + str(kx) + "$",
            #     )
            # for kz in range((Nz - 1) // 2 + 1)[0:10]:
            #     dots = ax_kz.plot(ts, amplitudes_2d_kzs_arr[:, kz], ".")
            #     ax_kz.plot(
            #         ts[: i + 1],
            #         amplitudes_2d_kzs_arr[: i + 1, kz],
            #         "o",
            #         color=dots[0].get_color(),
            #         label="$k_z = " + str(kz) + "$",
            #     )
            # ax_kx.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
            # ax_kx.set_box_aspect(1)
            # fname_kx = "plots/plot_amplitudes_kx_t_" + "{:06}".format(time_step)
            # # try:
            # #     fig_kx.savefig(
            # #         fname_kx + ".ps",
            # #         # fname_kx + ".png",
            # #         bbox_inches="tight",
            # #     )
            # #     psimage = Image.open(fname_kx + ".ps")
            # #     psimage.load(scale=10, transparency=True)
            # #     psimage.save(fname_kx + ".png", optimize=True)
            # #     image = Image.open(fname_kx + ".png")
            # #     imageBox = image.getbbox()
            # #     cropped = image.crop(imageBox)
            # #     cropped.save(fname_kx + ".png")
            # # except Exception:
            # fig_kx.savefig(
            #     # fname_kx + ".ps",
            #     fname_kx + ".png",
            #     bbox_inches="tight",
            # )

            # ax_kz.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
            # ax_kz.set_box_aspect(1)
            # fname_kz = "plots/plot_amplitudes_kz_t_" + "{:06}".format(time_step)
            # # try:
            # #     fig_kz.savefig(
            # #         fname_kz + ".ps",
            # #         # fname_kz + ".png",
            # #         bbox_inches="tight",
            # #     )
            # #     psimage = Image.open(fname_kz + ".ps")
            # #     psimage.load(scale=10, transparency=True)
            # #     psimage.save(fname_kz + ".png", optimize=True)
            # #     image = Image.open(fname_kz + ".png")
            # #     imageBox = image.getbbox()
            # #     cropped = image.crop(imageBox)
            # #     cropped.save(fname_kz + ".png")
            # # except Exception:
            # fig_kz.savefig(
            #     # fname_kz + ".ps",
            #     fname_kz + ".png",
            #     bbox_inches="tight",
            # )

            # ax_pd.plot(prod_arr, -diss_arr, "k.")
            # ax_pd.plot(
            #     np.array([0.0, max(-diss_arr)]),
            #     np.array([0.0, max(-diss_arr)]),
            #     color="0.8",
            #     linestyle="dashed",
            # )
            # ax_pd.plot(
            #     prod_arr[i],
            #     -diss_arr[i],
            #     "bo",
            # )
            # ax_pd.set_xlabel("$P$")
            # ax_pd.set_ylabel("$-D$")
            # fig_pd.savefig(
            #     Field.plotting_dir
            #     + "/plot_prod_diss_t_"
            #     + "{:06}".format(time_step)
            #     + ".png"
            # )

            # if i in [0, n_steps // 2, n_steps]:
            # if True:
            #     fig_kx = figure.Figure()
            #     ax_kx = fig_kx.subplots(1, 1)
            #     ax_kx_2 = ax_kx.twinx()
            #     fig_kz = figure.Figure()
            #     ax_kz = fig_kz.subplots(1, 1)
            #     ax_kz_2 = ax_kz.twinx()
            #     ax_kx.set_xscale("log")
            #     ax_kz.set_xscale("log")
            #     ax_kx_2.set_xscale("log")
            #     ax_kz_2.set_xscale("log")
            #     ax_kx.set_xlabel("$k_x$")
            #     ax_kz.set_xlabel("$k_z$")
            #     ax_kx.set_ylabel("$E$")
            #     ax_kz.set_ylabel("$E$")
            #     ax_kx_2.set_ylabel("$A$")
            #     ax_kz_2.set_ylabel("$A$")
            #     kxs = vel_hat.get_fourier_domain().grid[0]
            #     kzs = vel_hat.get_fourier_domain().grid[2]
            #     energy_kx = []
            #     energy_kz = []
            #     energy_x_kx = []
            #     energy_x_kz = []
            #     amp_kx = []
            #     amp_kz = []

            #     for kx in range((len(kxs) - 1) // 2):
            #         vel_2d_kx = vel_hat.field_2d(0, kx).no_hat()
            #         energy_kx.append(vel_2d_kx.energy())
            #         energy_x_kx.append(vel_2d_kx[0].energy())
            #         # amp_kx.append(vel_2d_kx[0].max() - vel_2d_kx[0].min())
            #         amp_kx.append(vel_2d_kx[0].inf_norm())

            #     for kz in range((len(kzs) - 1) // 2):
            #         vel_2d_kz = vel_hat.field_2d(2, kz).no_hat()
            #         energy_kz.append(vel_2d_kz.energy())
            #         energy_x_kz.append(vel_2d_kz[0].energy())
            #         # amp_kz.append(vel_2d_kz[0].max() - vel_2d_kz[0].min())
            #         amp_kz.append(vel_2d_kz[0].inf_norm())

            #     ax_kx.plot(energy_kx, "ko")
            #     ax_kz.plot(energy_kz, "ko")
            #     ax_kx.plot(energy_x_kx, "bo")
            #     ax_kz.plot(energy_x_kz, "bo")
            #     ax_kx_2.plot(amp_kx, "ro")
            #     ax_kz_2.plot(amp_kz, "ro")

            #     fig_kx.savefig(
            #         "plots/plot_energy_spectrum_kx_t_" + "{:06}".format(i) + ".png",
            #         bbox_inches="tight",
            #     )
            #     fig_kz.savefig(
            #         "plots/plot_energy_spectrum_kz_t_" + "{:06}".format(i) + ".png",
            #         bbox_inches="tight",
            #     )


def post_process_pub(
    file: str,
    Lx_over_pi: float,
    Lz_over_pi: float,
) -> None:

    n_snapshots = 3
    fig_pub_x_plane = figure.Figure(layout="tight", figsize=(15, 15))
    ax_pub_x_plane = fig_pub_x_plane.subplots(1, n_snapshots)
    fig_pub_z_plane = figure.Figure(layout="tight", figsize=(15, 15))
    ax_pub_z_plane = fig_pub_x_plane.subplots(1, n_snapshots)

    with h5py.File(file, "r") as f:
        velocity_trajectory = f["trajectory"]
        domain = get_domain(velocity_trajectory.shape[2:], Lx_over_pi, Lz_over_pi)
        n_fields = len(velocity_trajectory)
        n = 0
        for j in range(n_snapshots):
            i = (n_fields - 1) * j // (n_snapshots - 1)
            vel_hat = VectorField.FromData(
                FourierField, domain, velocity_trajectory[i], name="velocity_hat"
            )
            vel = vel_hat.no_hat()
            vel.set_time_step(j)
            vel.set_name("vel_pub")
            vel[0].plot_3d_single(
                0, name="$\\tilde{u}_x$", ax=ax_pub_x_plane[n], fig=fig_pub_x_plane
            )
            vel[0].plot_3d_single(
                2, name="$\\tilde{u}_x$", ax=ax_pub_z_plane[n], fig=fig_pub_z_plane
            )
            vel[0].plot_3d_single(0, name="$\\tilde{u}_x$")
            vel[0].plot_3d_single(2, name="$\\tilde{u}_x$")
            n += 1
    fig_pub_x_plane.savefig(
        "plots/vel_pub_x_plane.png",
        bbox_inches="tight",
    )
    fig_pub_z_plane.savefig(
        "plots/vel_pub_z_plane.png",
        bbox_inches="tight",
    )


# args = {}
assert len(sys.argv) > 1, "please provide a trajectory file to analyse"
assert (
    len(sys.argv) <= 2
), "there is no need to provide further arguments as these are inferred automatically from simulation_settings.yml"

# post_process_pub(
#     sys.argv[1],
#     args.get("Lx_over_pi", 1.0),
#     args.get("Lz_over_pi", 1.0),
# )

post_process(
    sys.argv[1],
    args.get("end_time", 0.7),
    args.get("Lx_over_pi", 1.0),
    args.get("Lz_over_pi", 1.0),
    args.get("Re_tau", 180.0),
    0,
)

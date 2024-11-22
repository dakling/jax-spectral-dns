#!/usr/bin/env python3

import time
import jax

from jax_spectral_dns.equation import Equation, print_verb

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from enum import Enum
from typing import Any, Tuple, Optional, List, Self, cast
import numpy as np
import numpy.typing as npt
import yaml
import os
from glob import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.patches as mpatches

from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.field import VectorField, PhysicalField

matplotlib.set_loglevel("error")
matplotlib.use("ps")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)


class Case:
    STORE_DIR_BASE = (
        "/store/DAMTP/dsk34/"
        if os.getenv("HOSTNAME") and "maths.cam.ac.uk" in os.getenv("HOSTNAME")
        else "/home/klingenberg/mnt/maths_store/"
    )
    HOME_DIR_BASE = (
        "/home/dsk34/jax-optim/run/"
        if os.getenv("HOSTNAME") and "maths.cam.ac.uk" in os.getenv("HOSTNAME")
        else "/home/klingenberg/mnt/maths/jax-optim/run/"
    )
    Vel_0_types = Enum(
        "vel_0_types", ["quasilinear", "nonlinear_global", "nonlinear_localised"]
    )

    def __init__(self, directory: str):
        self.directory = directory
        self.T = self.get_T()
        self.e_0 = self.get_e0()
        self.gain = self.get_gain()
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

    @classmethod
    def sort_by_e_0(cls, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.e_0)
        return cases

    def get_domain(self) -> "PhysicalDomain":
        Nx = cast(int, self.get_property_from_settings("Nx", 48))
        Ny = cast(int, self.get_property_from_settings("Ny", 129))
        Nz = cast(int, self.get_property_from_settings("Nz", 80))
        sc_x = cast(float, self.get_property_from_settings("Lx_over_pi", 2.0))
        sc_z = cast(float, self.get_property_from_settings("Lx_over_pi", 2.0))
        domain = PhysicalDomain.create(
            (Nx, Ny, Nz), (True, False, True), (sc_x, 1.0, sc_z)
        )
        return domain

    def get_vel_0(self) -> "VectorField[PhysicalField]":
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
                        domain, path + file, name="velocity_pert", allow_projection=True
                    )
                    success = True
                except Exception:
                    try:
                        bak_file = glob("velocity_latest_bak_*", root_dir=path)[0]
                        vel_0 = VectorField.FromFile(
                            domain, path + bak_file, name="vel_0", allow_projection=True
                        )
                        success = True
                    except Exception:
                        try:
                            bak_file = glob("velocity_latest_bak_*", root_dir=path)[0]
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
        return vel_0

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
            return "."
        elif vel_0_type is Case.Vel_0_types["nonlinear_localised"]:
            print_verb("nl loc", verbosity_level=3)
            return "o"
        else:
            print(self)
            print(self.vel_0_type)
            raise Exception("unknown velocity classification.")

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

    def append_to(self, cases: "List[Self]") -> "List[Self]":
        if self.successfully_read:
            assert self.gain is not None
            Ts = [case.T for case in cases]
            e_0s = [case.e_0 for case in cases]
            gains = [case.gain for case in cases]
            if (self.T in Ts) and (self.e_0 in e_0s):
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
    def collect(cls, base_path: str) -> "List[Case]":
        home_path = Case.HOME_DIR_BASE + "/" + base_path
        dirs = glob("[0-9]eminus[0-9]", root_dir=home_path)
        dirs += glob("[0-9]eminus[0-9]_sweep_*", root_dir=home_path)
        dirs += glob("[0-9]eminus[0-9]_from_*", root_dir=home_path)
        cases = []
        for dir in dirs:
            case = Case(base_path + "/" + dir)
            cases = case.append_to(cases)
        case = Case(base_path + "/" + "linear")
        case.e_0 = 0.0
        cases = case.append_to(cases)
        Case.sort_by_e_0(cases)
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


def plot(dirs_and_names: List[str]) -> None:
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    ax.set_xlabel("$T h  / u_\\tau$")
    ax.set_ylabel("$e_0/E_0$")
    ax.set_yscale("log")
    e_0_lam_boundary = []
    e_0_nl_lower_glob_boundary = []
    e_0_nl_upper_glob_boundary = []
    e_0_nl_lower_loc_boundary = []
    e_0_nl_upper_loc_boundary = []
    Ts = []
    all_cases = []
    for base_path, name, _ in dirs_and_names:
        print_verb("collecting cases in", base_path)
        cases = Case.collect(base_path)
        cases = Case.prune_times(cases)  # TODO why is this necessary?
        all_cases += cases
        print_verb(
            "collected cases:",
            "; ".join([case.directory.split("/")[-1] for case in cases]),
        )
        Ts.append(cases[0].T)
        # max_i = (
        #     np.argmax([cast(float, case.gain) for case in cases[1:]]) + 1
        # )  # TODO exclude truly laminar case?
        e_0_lam_boundary.append(Case.get_e_0_lam_upper_boundary(cases))
        e_0_nl_lower_glob_boundary.append(Case.get_e_0_nl_glob_lower_boundary(cases))
        e_0_nl_upper_glob_boundary.append(Case.get_e_0_nl_glob_upper_boundary(cases))
        e_0_nl_lower_loc_boundary.append(Case.get_e_0_nl_loc_lower_boundary(cases))
        e_0_nl_upper_loc_boundary.append(Case.get_e_0_nl_loc_upper_boundary(cases))
        for i in range(len(cases)):
            print_verb(cases[i], verbosity_level=3)
            marker = cases[i].get_marker()
            name = cases[i].get_classification_name()
            # color = "r" if i == max_i else "k"
            color = "k"
            ax.plot(
                cast(float, cases[i].T),
                cast(float, cases[i].e_0),
                color + marker,
                label=name,
            )

    # plot isosurfaces of gain
    try:
        tcf = ax.tricontourf(
            np.array([c.T for c in all_cases]),
            np.array([c.e_0 for c in all_cases]),
            np.array([c.gain for c in all_cases]),
            20,
            shading="gouraud",
            locator=matplotlib.ticker.LogLocator(),
        )
    except Exception as e:
        print("plotting with gouraud failed:")
        print(e)
        tcf = ax.tricontourf(
            np.array([c.T for c in all_cases]),
            np.array([c.e_0 for c in all_cases]),
            np.array([c.gain for c in all_cases]),
            20,
            locator=matplotlib.ticker.LogLocator(),
        )
    fig.colorbar(tcf)
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))
    fig.savefig("plots/T_e_0_space.png")
    # # paint linear regime dark grey
    try:
        ax.fill_between(
            Ts,
            [0 for _ in range(len(e_0_lam_boundary))],
            e_0_lam_boundary,
            color="grey",
            alpha=0.7,
            interpolate=True,
        )
    except Exception:
        print("drawing linear regime did not work")
        pass
    # try:
    #     ax.fill_between(
    #         [
    #             Ts[i]
    #             for i in range(len(Ts))
    #             if e_0_nl_upper_glob_boundary[i] is not None
    #         ],
    #         [
    #             e_0_lam_boundary[i]
    #             for i in range(len(Ts))
    #             if e_0_nl_upper_glob_boundary[i] is not None
    #         ],
    #         [
    #             e_0_nl_lower_glob_boundary[i]
    #             for i in range(len(Ts))
    #             if e_0_nl_upper_glob_boundary[i] is not None
    #         ],
    #         color="grey",
    #         alpha=0.55,
    #         interpolate=True,
    #     )
    # except Exception:
    #     print("drawing intermediate regime did not work")
    #     pass
    # paint nonlinear global regime light grey
    try:
        ax.fill_between(
            [
                Ts[i]
                for i in range(len(Ts))
                if e_0_nl_upper_glob_boundary[i] is not None
            ],
            [
                e_0_nl_lower_glob_boundary[i]
                for i in range(len(Ts))
                if e_0_nl_upper_glob_boundary[i] is not None
            ],
            [
                e_0_nl_upper_glob_boundary[i]
                for i in range(len(Ts))
                if e_0_nl_upper_glob_boundary[i] is not None
            ],
            color="grey",
            alpha=0.5,
            interpolate=True,
        )
    except Exception:
        print("drawing nonlinear global regime did not work")
        pass
    [
        e_0_nl_lower_loc_boundary[i]
        for i in range(len(Ts))
        if e_0_nl_upper_loc_boundary[i] is not None
    ],
    [
        e_0_nl_upper_loc_boundary[i]
        for i in range(len(Ts))
        if e_0_nl_upper_loc_boundary[i] is not None
    ],
    try:
        ax.fill_between(
            [Ts[i] for i in range(len(Ts)) if e_0_nl_upper_loc_boundary[i] is not None],
            [
                e_0_nl_lower_loc_boundary[i]
                for i in range(len(Ts))
                if e_0_nl_upper_loc_boundary[i] is not None
            ],
            [
                e_0_nl_upper_loc_boundary[i]
                for i in range(len(Ts))
                if e_0_nl_upper_loc_boundary[i] is not None
            ],
            color="grey",
            alpha=0.3,
            interpolate=True,
        )
    except Exception:
        print("drawing nonlinear localised regime did not work")
        pass
    ax.text(1.35, 3.0e-6, "unexplored")
    ax.text(1.25, 1.7e-5, "unexplored")
    ax.text(0.5, 2.0e-6, "quasi-linear regime")
    ax.text(1.95, 1.6e-5, "nonlinear global regime")
    ax.text(1.6, 7.0e-5, "nonlinear \n localised \n regime")
    # ax.text(
    #     1.4,
    #     6.0e-4,
    #     "no convergence due to \n turbulent end state",
    #     backgroundcolor="white",
    # )
    # handles, labels = ax.get_legend_handles_labels()
    # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # ax.legend(*zip(*unique))
    fig.savefig("plots/T_e_0_space_with_regimes.png")


e_base_turb = 1.0
e_base_lam = 2160.0 / 122.756

dirs_and_names = [
    (
        "smaller_channel_one_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=0.35 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_one_pt_five_t_0_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=0.525 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_two_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=0.7 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_three_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=1.05 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_four_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=1.4 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_six_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=2.1 h / u_\\tau$",
        e_base_turb,
    ),
    (
        "smaller_channel_eight_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=2.8 h / u_\\tau$",
        e_base_turb,
    ),
]

if __name__ == "__main__":
    plot(dirs_and_names)

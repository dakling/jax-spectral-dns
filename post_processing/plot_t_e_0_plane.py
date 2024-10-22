#!/usr/bin/env python3


from typing import Tuple, Optional, List, Self
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.figure as figure

import matplotlib

matplotlib.set_loglevel("error")
matplotlib.use("ps")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)


class Case:
    STORE_DIR_BASE = "/home/klingenberg/mnt/maths_store/"
    HOME_DIR_BASE = "/home/klingenberg/mnt/maths/jax-optim/run/"

    def __init__(self, directory: str):
        self.T = self.get_T(self.HOME_DIR_BASE, directory)
        self.e_0 = self.get_e0(self.HOME_DIR_BASE, directory)
        self.gain = self.get_gain(self.STORE_DIR_BASE, directory)
        self.successfully_read = (
            self.T is not None and self.e_0 is not None and self.gain is not None
        )
        self.directory = directory

    def get_property_from_settings(
        self, base_path: str, directory: str, property: str
    ) -> Optional[float]:
        fname = base_path + "/" + directory + "/simulation_settings.yml"
        try:
            with open(fname, "r") as file:
                args = yaml.safe_load(file)
            return args[property]
        except Exception as e:
            return None

    def get_e0(self, base_path: str, directory: str) -> Optional[float]:
        return self.get_property_from_settings(base_path, directory, "e_0")

    def get_T(self, base_path: str, directory: str) -> Optional[float]:
        return self.get_property_from_settings(base_path, directory, "end_time")

    def get_gain(self, base_path, directory: str) -> Optional[float]:
        phase_space_data_name = (
            base_path + "/" + directory + "/plots/phase_space_data.txt"
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

    def sort_by_e_0(self, cases: "List[Self]") -> "List[Self]":
        cases.sort(key=lambda x: x.e_0)
        return cases

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
    def collect(cls, base_path: str) -> "List[Case]":
        home_path = Case.HOME_DIR_BASE + "/" + base_path
        dirs = glob("[0-9]eminus[0-9]", root_dir=home_path)
        dirs += glob("[0-9]eminus[0-9]_sweep_down", root_dir=home_path)
        dirs += glob("[0-9]eminus[0-9]_from_linopt", root_dir=home_path)
        dirs += glob("[0-9]eminus[0-9]_from_lin_opt", root_dir=home_path)
        cases = []
        for dir in dirs:
            case = Case(base_path + "/" + dir)
            cases = case.append_to(cases)
        case = Case(base_path + "/" + "linear")
        case.e_0 = 0.0
        cases = case.append_to(cases)
        case.sort_by_e_0(cases)
        return cases

    @classmethod
    def get_e_0_lam_boundary(cls, cases: "List[Case]") -> "Tuple[float, int]":
        e_0_lam_boundary_change = True
        e_0_lam_boundary = 0.0
        j = 0
        for i in range(len(cases)):
            g_lin = cases[0].gain
            g = cases[i].gain
            assert g_lin is not None
            assert g is not None
            if g <= g_lin * 1.05 and e_0_lam_boundary_change:
                e_0_lam_boundary = cases[i].e_0
                j = i
            else:
                e_0_lam_boundary_change = False
        assert e_0_lam_boundary is not None
        return e_0_lam_boundary, j


def plot(dirs_and_names: List[str]) -> None:
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    ax.set_xlabel("$T h  / u_\\tau$")
    ax.set_ylabel("$e_0/E_0$")
    ax.set_yscale("log")
    e_0_lam_boundary = []
    Ts = []
    for base_path, name in dirs_and_names:
        cases = Case.collect(base_path)
        Ts.append(cases[0].T)
        max_i = np.argmax([case.gain for case in cases])
        e_0_lam_boundary_, k = Case.get_e_0_lam_boundary(cases)
        e_0_lam_boundary.append(e_0_lam_boundary_)
        for i in range(len(cases)):
            if i <= k:
                marker = "x"
            else:
                marker = "o"
            color = (
                "r" if i == max_i else "k"
            )  # TODO encode information in color -> maximium gain at this time
            ax.plot(cases[i].T, cases[i].e_0, color + marker)
    # paint linear regime grey
    ax.fill_between(
        Ts,
        [0 for _ in range(len(e_0_lam_boundary))],
        e_0_lam_boundary,
        color="grey",
        alpha=0.5,
        interpolate=True,
    )
    ax.text(1.5, 0.9e-6, "linear regime", backgroundcolor="white")
    ax.text(1.5, 3.0e-5, "nonlinear regime", backgroundcolor="white")
    fig.savefig("plots/T_e_0_space.png")


dirs_and_names = [
    (
        "smaller_channel_one_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=0.35 h / u_\\tau$",
    ),
    (
        "smaller_channel_two_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=0.7 h / u_\\tau$",
    ),
    (
        "smaller_channel_three_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=1.05 h / u_\\tau$",
    ),
    (
        "smaller_channel_four_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=1.4 h / u_\\tau$",
    ),
    (
        "smaller_channel_six_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=2.1 h / u_\\tau$",
    ),
    (
        "smaller_channel_eight_t_e_0_study",
        # "minimal channel mean (short channel)",
        "$T=2.8 h / u_\\tau$",
    ),
]

plot(dirs_and_names)

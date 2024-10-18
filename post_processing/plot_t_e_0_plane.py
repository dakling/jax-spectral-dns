#!/usr/bin/env python3


from typing import Tuple, Optional, List
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

STORE_DIR_BASE = "/home/klingenberg/mnt/maths_store/"
HOME_DIR_BASE = "/home/klingenberg/mnt/maths/jax-optim/run/"


def get_property_from_settings(base_path: str, directory: str, property: str) -> float:
    fname = base_path + "/" + directory + "/simulation_settings.yml"
    with open(fname, "r") as file:
        args = yaml.safe_load(file)
    return args[property]


def get_e0(base_path: str, directory: str) -> float:
    return get_property_from_settings(base_path, directory, "e_0")


def get_T(base_path: str, directory: str) -> float:
    return get_property_from_settings(base_path, directory, "end_time")


def get_gain(base_path, directory: str) -> Optional[float]:
    phase_space_data_name = base_path + "/" + directory + "/plots/phase_space_data.txt"
    phase_space_data = np.atleast_2d(
        np.genfromtxt(
            phase_space_data_name,
            delimiter=",",
        )
    ).T
    return max(phase_space_data[1])


def collect(
    home_path: str, store_path: str
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dirs = glob("[0-9]eminus[0-9]", root_dir=home_path)
    e_0s = []
    Ts = []
    gains = []
    for dir in dirs:
        try:
            e_0 = get_e0(home_path, dir)
            T = get_T(home_path, dir)
            gain = get_gain(store_path, dir)
        except Exception as e:
            e_0 = None
            T = None
            gain = None
        if e_0 is not None and T is not None and gain is not None:
            e_0s.append(e_0)
            Ts.append(T)
            gains.append(gain)
    try:
        gain = get_gain(store_path, "linear")
        T = get_T(home_path, "linear")
    except Exception as e:
        gain = None
        T = None
    if gain is not None and T is not None:
        e_0s.append(0.0)
        Ts.append(T)
        gains.append(gain)
    Ts, e_0s, gains = (list(x) for x in zip(*sorted(zip(Ts, e_0s, gains))))
    return np.array(Ts), np.array(e_0s), np.array(gains)


def plot(dirs_and_names: List[str]) -> None:
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    ax.set_xlabel("$T h  / u_\\tau$")
    ax.set_ylabel("$e_0/E_0$")
    ax.set_yscale("log")
    e_0_lam_boundary = [0.0 for _ in range(len(dirs_and_names))]
    j = 0
    Ts = []
    for base_path, name in dirs_and_names:
        store_dir_base = STORE_DIR_BASE + "/" + base_path
        home_dir_base = HOME_DIR_BASE + "/" + base_path
        T, e_0, gain = collect(home_dir_base, store_dir_base)
        Ts.append(T[0])
        max_i = np.argmax(gain)
        for i in range(len(T)):
            g_lin = gain[0]
            g = gain[i]
            e_0_lam_boundary_change = True
            if g <= g_lin * 1.05 and e_0_lam_boundary_change:
                marker = "x"
                e_0_lam_boundary[j] = e_0[i]
            else:
                marker = "o"
                e_0_lam_boundary_change = False
            color = (
                "r" if i == max_i else "k"
            )  # TODO encode information in color -> maximium gain at this time
            ax.plot(T[i], e_0[i], color + marker)
        j += 1
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

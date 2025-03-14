#!/usr/bin/env python3

from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

from jax_spectral_dns.equation import print_verb
from plot_t_e_0_plane import Case, dirs_and_names

matplotlib.set_loglevel("error")
matplotlib.use("ps")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
        "font.size": 18,
    }
)


def get_data(path: str) -> "Tuple(np_array, np_array)":
    base_path = "/home/klingenberg/mnt/maths_store/"
    data_x = np.genfromtxt(base_path + path + "/plots/amplitudes_x.txt")
    # highest_kx = 8
    highest_kx = 6
    ts = data_x[:, 0]
    amp_xs = data_x[:, 1 : (highest_kx // 2 + 2)].T
    return (ts, amp_xs)


def plot_single(fig, ax, dir, name, mark):
    t, amp_xs = get_data(dir)
    k = 0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for amp_x, color in zip(amp_xs, colors):
        ax.plot(
            t,
            amp_x,
            # mark,
            "-",
            label="$k_x = " + str(k) + "$" + (" (streaks)" if k == 0 else ""),
            color=color,
        )
        k += 2


def plot(dirs_and_names):
    # TODO try subplots

    # fig = matplotlib.pyplot.figure(figsize=(12, 12))
    fig = matplotlib.pyplot.figure(figsize=(9, 6 * 9 / 8))
    ax = fig.subplots(len(dirs_and_names), 1)
    if len(dirs_and_names) == 1:
        ax = [ax]
    i = 0
    ax[-1].set_xlabel("$t u_\\tau / h$")
    offset = 0.01
    # ypos = [0.5, 0.2, 0.5]
    ypos = [0.2]
    linewd = 0.9 * matplotlib.rcParams["lines.linewidth"]
    for dir, name, mark in dirs_and_names:
        ax[i].set_ylabel("$|\\tilde{u}|_\\infty$")
        ax[i].set_yscale("log")
        plot_single(fig, ax[i], dir, name, mark)
        ax[i].vlines(
            0.08, ymin=1e-2, ymax=1e1, linestyle=":", color="k", linewidth=linewd
        )
        # ax[i].vlines(0.23, ymin=1e-2,  ymax=1e1, linestyle=":", color="k", linewidth=linewd)
        ax[i].vlines(
            0.5, ymin=1e-2, ymax=1e1, linestyle=":", color="k", linewidth=linewd
        )
        # fig.text(0.08 + offset, ypos[i], '(a)', transform=ax[i].get_xaxis_transform())
        # fig.text(0.23 + offset, ypos[i], '(b)', transform=ax[i].get_xaxis_transform())
        # fig.text(0.6 + offset, ypos[i], '(c)', transform=ax[i].get_xaxis_transform())
        i += 1

    if len(dirs_and_names) == 1:
        fig.legend(loc="upper left")
        # fig.legend(loc="lower center")
    fig.savefig("amp_x_over_time.png", bbox_inches="tight")


# dirs_and_names = [
#     (
#         "smaller_channel_four_t_e_0_study/3eminus5",
#         "$T=1.4 h / u_\\tau; e_0 / E_0 = 3 \\times 10^{-5} $",
#     ),
#     (
#         "smaller_channel_six_t_e_0_study/1eminus4",
#         "$T=2.1 h / u_\\tau; e_0 / E_0 = 1 \\times 10^{-4} $",
#     ),
#     (
#         "smaller_channel_six_t_e_0_study/3eminus5",
#         "$T=2.1 h / u_\\tau; e_0 / E_0 = 3 \\times 10^{-5} $",
#     ),
#     (
#         "smaller_channel_eight_t_e_0_study/3eminus5",
#         "$T=2.8 h / u_\\tau; e_0 / E_0 = 3 \\times 10^{-5} $",
#     ),
# ]
dirs_and_names = [
    # (
    #     "smaller_channel_two_t_e_0_study/3eminus5",
    #     "$T=0.7 h / u_\\tau$ (localised)",
    #     ":"
    # ),
    (
        "smaller_channel_two_t_e_0_study/1eminus4_sweep_down",
        "$T=0.7 h / u_\\tau$ (localised)",
        "--",
    ),
    # (
    #     "smaller_channel_two_t_e_0_study/1eminus4",
    #     "$T=0.7 h / u_\\tau$ (localised)",
    #     "-"
    # ),
]

plot(dirs_and_names)

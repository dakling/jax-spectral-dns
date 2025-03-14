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
    data = np.genfromtxt(base_path + path + "/plots/energy.txt")
    ts = data[:, 0]
    energy = data[:, 1]
    return (ts, energy)


def plot_single(fig, ax, dir, name, mark):
    t, e = get_data(dir)
    ax.plot(t, e / e[0], "-", label=name)


def plot(dirs_and_names):

    fig = matplotlib.pyplot.figure()
    ax = fig.subplots(1, 1)
    ax.set_xlabel("$t u_\\tau / h$")
    ax.set_ylabel("$G$")
    for dir, name, mark in dirs_and_names:
        plot_single(fig, ax, dir, name, mark)
    # ax.vlines(0.7,ymin=0,  ymax=160, linestyle="--", color="k", label="$t = T = 0.7 h / u_\\tau$")

    # fig.legend(loc="upper left")
    fig.savefig("gains_over_time.png", bbox_inches="tight")


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
    (
        "smaller_channel_four_t_e_0_study/3eminus5",
        "$T=1.4 h / u_\\tau$ (localised)",
        "o",
    ),
    (
        "smaller_channel_six_t_e_0_study/3eminus5",
        "$T=2.1 h / u_\\tau$ (not localised)",
        "s",
    ),
    (
        "smaller_channel_eight_t_e_0_study/3eminus5",
        "$T=2.8 h / u_\\tau$ (not localised)",
        "s",
    ),
]

plot(dirs_and_names)

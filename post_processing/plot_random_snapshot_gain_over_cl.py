#!/usr/bin/env python3


from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib

from jax_spectral_dns.equation import print_verb
from plot_t_e_0_plane import Case, dirs_and_names

matplotlib.set_loglevel("error")
matplotlib.use("ps")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)


def plot():
    fig, ax = plt.subplots(3, 1, figsize=(9, 12))
    fig.subplots_adjust()  # adjust space between Axes

    colors = ["b", "k", "r", "c", "m", "y"]
    base_path = "random_mean_snapshot/"
    print_verb("collecting cases in", base_path)
    col = colors.pop(0)
    # cases = Case.collect(base_path, prune_duplicates=False, with_linear=False, legal_names=["[0-9]"])
    cases = Case.collect(
        base_path, prune_duplicates=False, with_linear=False, legal_names=["*"]
    )
    cases = Case.sort_by_u_wall(cases)
    print_verb(
        "collected cases:",
        "; ".join([case.directory.split("/")[-1] for case in cases]),
    )
    # ax.plot([case.get_base_velocity_cl() for case in cases], [case.gain for case in cases], "o", color=col, label="$T = " + str(cases[0].T) + " h / u_\\tau$")
    n_cases = len(cases)
    for i, c in enumerate(cases):
        ax[0].plot(
            [c.get_base_velocity_max()],
            [c.gain],
            "o",
            color=col,
            alpha=(i + 1) / n_cases,
            label="$T = " + str(cases[0].T) + " h / u_\\tau$",
        )
        ax[1].plot(
            [c.get_base_velocity_wall()],
            [c.gain],
            "o",
            color=col,
            alpha=(i + 1) / n_cases,
            label="$T = " + str(cases[0].T) + " h / u_\\tau$",
        )
        ax[2].plot(
            [c.get_base_velocity_max()],
            [c.get_base_velocity_wall()],
            "o",
            color=col,
            alpha=(i + 1) / n_cases,
            label="$T = " + str(cases[0].T) + " h / u_\\tau$",
        )
    # ax.plot(xs, ys, "b-", label="ad-hoc cos fit")
    # ax.axvline(x=x_0+delta/2, color="y",  label="cos fit inflection point")
    # ax.axvline(x=cases[6].get_base_velocity_cl(), color="g", label="u_max_mean")
    ax[0].set_ylim(ymin=1.0)
    ax[1].set_ylim(ymin=1.0)
    # ax_.set_xlim([-1e-20, 1e-20])
    # ax.set_xlabel("histogram bin")
    ax[0].set_xlabel("$U_\\text{max}$")
    ax[0].set_ylabel("$G_\\text{opt}$")

    ax[1].set_xlabel("$U_\\text{wall}$")
    ax[1].set_ylabel("$G_\\text{opt}$")

    ax[2].set_xlabel("$U_\\text{max}$")
    ax[2].set_ylabel("$U_\\text{wall}$")
    fig.tight_layout()
    # fig.legend(loc="upper left")
    fig.savefig("plots/plot_random_snapshot_gains.png")
    print("done")


plot()

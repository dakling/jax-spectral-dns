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


def plot(base_paths):
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust()  # adjust space between Axes

    colors = ["b", "k", "r", "c", "m", "y"]
    for base_path in base_paths:
        print_verb("collecting cases in", base_path)
        col = colors.pop(0)
        cases = Case.collect(
            base_path, prune_duplicates=False, with_linear=False, legal_names=["*"]
        )
        cases = Case.sort_by_u_base_cl(cases)
        print_verb(
            "collected cases:",
            "; ".join([case.directory.split("/")[-1] for case in cases]),
        )
        ax.plot(
            [case.get_base_velocity_cl() for case in cases],
            [case.gain for case in cases],
            "--",
            color=col,
        )
        ax.plot(
            [case.get_base_velocity_cl() for case in cases],
            [case.gain for case in cases],
            "o",
            color=col,
            label="$T = " + str(cases[0].T) + " h / u_\\tau$",
        )
        # ax.plot([case.get_base_velocity_wall() for case in cases], [case.gain for case in cases], "--", color=col)
        # ax.plot([case.get_base_velocity_wall() for case in cases], [case.gain for case in cases], "o", color=col)
        x_0 = 17
        delta = 3.5
        y_min = 58
        y_max = 172
        xs = np.arange(x_0, x_0 + delta, 0.01)
        ys = (y_max - y_min) * (
            1 - np.cos(1.0 * np.pi / delta * (xs - x_0))
        ) / 2 + y_min
    # ax.plot(xs, ys, "b-", label="ad-hoc cos fit")
    # ax.axvline(x=x_0+delta/2, color="y",  label="cos fit inflection point")
    # ax.axvline(x=cases[6].get_base_velocity_cl(), color="g", label="u_max_mean")
    ax.axvline(x=18.1, color="g", label="u_max_mean")
    ax.set_ylim(ymin=1.0)
    # ax_.set_xlim([-1e-20, 1e-20])
    # ax.set_xlabel("histogram bin")
    ax.set_xlabel("$U_\\text{cl}$")
    ax.set_ylabel("$G_\\text{opt}$")
    fig.tight_layout()
    fig.legend(loc="upper left")
    fig.savefig("plots/plot_hist_gains.png")
    print("done")


# plot(["hist_9_study_two_time_units/", "hist_9_study_three_time_units/", "hist_9_study/"])
plot(
    [
        "hist_18_study_three_time_units/",
        "hist_9_study_two_time_units/",
        "hist_9_study_three_time_units/",
        "hist_9_study/",
    ]
)

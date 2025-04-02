#!/usr/bin/env python3


from enum import Enum
from typing import Tuple, Optional
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.patches as mpatches

from jax_spectral_dns.equation import print_verb
from plot_t_e_0_plane import Case, CessCase, PertCase, dirs_and_names

matplotlib.set_loglevel("error")
matplotlib.use("ps")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{amsmath}" "\\usepackage{xcolor}",
    }
)

# grad = True
grad = False


class CaseGroup:

    artificial_types = Enum("artificial_types", ["Cess", "Pert", "None"])

    def __init__(self, path, color, artificial_type):
        self.color = color
        self.path = path
        self.artificial_type = CaseGroup.artificial_types[artificial_type]
        self.cases = None

    def __repr__(self):
        return self.path

    def get_marker(self):
        if self.artificial_type is CaseGroup.artificial_types["None"]:
            return "o"
        elif self.artificial_type is CaseGroup.artificial_types["Cess"]:
            return "x"
        elif self.artificial_type is CaseGroup.artificial_types["Pert"]:
            return "x"
        else:
            raise Exception(
                "unknown case artificiality type " + str(self.artificial_type)
            )

    def collect(self, prune_duplicates=False, with_linear=False, legal_names=["*"]):
        if self.artificial_type is CaseGroup.artificial_types["None"]:
            self.cases = Case.collect(
                self.path,
                prune_duplicates=prune_duplicates,
                with_linear=with_linear,
                legal_names=legal_names,
            )
        elif self.artificial_type is CaseGroup.artificial_types["Cess"]:
            self.cases = CessCase.collect(
                self.path,
                prune_duplicates=prune_duplicates,
                with_linear=with_linear,
                legal_names=legal_names,
            )
        elif self.artificial_type is CaseGroup.artificial_types["Pert"]:
            self.cases = PertCase.collect(
                self.path,
                prune_duplicates=prune_duplicates,
                with_linear=with_linear,
                legal_names=legal_names,
            )
        else:
            raise Exception(
                "unknown case artificiality type " + str(self.artificial_type)
            )

    def sort_by_u_base_cl(self):
        self.cases = Case.sort_by_u_base_cl(self.cases)

    @classmethod
    def flatten(cls, case_groups):
        return [x for xs in case_groups for x in xs.cases]


def get_correlation_quality(case_groups, y, ax):
    cases_flat = CaseGroup.flatten(case_groups)
    if not grad:
        y_s = np.array([case.get_base_velocity_at_pm_y(y) for case in cases_flat])
    else:
        y_s = np.array(
            [case.get_base_velocity_diff_y_at_pm_y(y) for case in cases_flat]
        )
    gain_s = np.array([case.get_gain() for case in cases_flat])
    if not grad:
        p, residuals, rank, singular_values, rcond = np.polyfit(
            y_s, gain_s, deg=1, full=True
        )
    else:
        p, residuals, rank, singular_values, rcond = np.polyfit(
            y_s, gain_s, deg=1, full=True
        )
    start_index = 0
    for css in case_groups:
        end_index = len(css.cases) + start_index
        if not grad:
            y_s_ = y_s[start_index:end_index]
        else:
            y_s_ = y_s[start_index:end_index]
        gain_s_ = gain_s[start_index:end_index]
        start_index = end_index
        if not grad:
            ax.plot(y_s_, gain_s_, css.color + css.get_marker())
        else:
            ax.plot(y_s_, gain_s_, css.color + css.get_marker())
    if not grad:
        ax.plot(y_s, [vel * p[0] + p[1] for vel in y_s], "k-")
    else:
        ax.plot(y_s, [vel * p[0] + p[1] for vel in y_s], "k-")
    ax.set_title("$y = " + str(y) + "$")
    ax.set_xlabel("$U(y=" + str(y) + ")$")
    ax.set_ylabel("$G$")
    fname = "plots/plot_corr_y" + ("_grad_y" if grad else "") + str(y)
    fname = fname.replace(".", "_dot_")
    fname = fname.replace("-", "_minus_")
    fname = fname + ".png"
    return residuals[0]


def plot(groups):
    cases = []
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust()  # adjust space between Axes

    for group in groups:
        base_path = group.path
        print_verb("collecting cases in", base_path)
        group.collect(prune_duplicates=False, with_linear=False, legal_names=["*"])
        cases.append(group)
        group.sort_by_u_base_cl()
        print_verb(
            "collected cases:",
            "; ".join([case.directory.split("/")[-1] for case in group.cases]),
        )
    ys = [
        0,
        0.5,
        0.53,
        0.55,
        0.57,
        0.58,
        0.6,
        0.7,
        0.75,
        0.8,
        0.81,
        0.82,
        0.84,
        0.85,
        0.9,
        0.95,
    ]
    residuals = []

    fig_corr, ax_corr = plt.subplots(len(ys), 1, figsize=(6, len(ys) * 4))
    for i, y in enumerate(ys):
        print_verb("Doing y =", y)
        fig_, ax_ = plt.subplots(1, 1)
        # residuals.append(get_correlation_quality(cases, y, ax_corr[i]))
        residuals.append(get_correlation_quality(cases, y, ax_))
        print_verb("res =", residuals[-1])
        fig_.tight_layout()
        # fig_.savefig("plots/plot_corrs_" + str(i) + ".png")

        try:
            fig_.legend(
                loc="upper left",
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=cases[i].color,
                        marker=cases[i].get_marker(),
                        linestyle="",
                        label=base_paths[i],
                    )
                    for i in range(len(base_paths))
                ],
            )
        except Exception as e:
            print(e)
        if grad:
            fig_.savefig("plots/plot_corrs_grad_" + str(i) + ".png")
        else:
            fig_.savefig("plots/plot_corrs_" + str(i) + ".png")
    print(ys)
    print(residuals)
    ax.set_xlabel("$y$")
    ax.set_ylabel("residual")
    ax.plot(ys, residuals)

    fig.tight_layout()
    fig.savefig("plots/plot_y_correlation.png")
    try:
        fig_corr.legend(
            loc="upper left",
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=cases[i].color,
                    marker=cases[i].get_marker(),
                    linestyle="",
                    label=base_paths[i],
                )
                for i in range(len(base_paths))
            ],
        )
    except Exception as e:
        print(e)
    fig_corr.tight_layout()
    fig_corr.savefig("plots/plot_corrs.png")
    print_verb("done")


colors = ["y", "b", "r", "c", "m", "k"]
base_paths = [
    # CaseGroup("cess_three_time_units", colors.pop(), "Cess"),
    # CaseGroup("base_variation_three_time_units", colors.pop(), "Pert"),
    CaseGroup("random_mean_snapshot/", colors.pop(0), "None"),
    CaseGroup("hist_18_study_three_time_units/", colors.pop(0), "None"),
    CaseGroup("hist_9_study_three_time_units/", colors.pop(0), "None"),
]
plot(base_paths)

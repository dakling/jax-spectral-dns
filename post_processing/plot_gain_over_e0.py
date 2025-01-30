#!/usr/bin/env python3

from typing import Tuple, Optional
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
        "font.size": 20,
    }
)


def plot_single(
    fig, ax, ax_, base_path: str, name: str, e_base: float = 1.0, rel: bool = False
) -> None:
    try:
        print_verb("collecting cases in", base_path)
        cases = Case.collect(base_path)
        cases = Case.sort_by_e_0(cases)
        cases = Case.prune_times(cases)
        print_verb(
            "collected cases:",
            "; ".join([case.directory.split("/")[-1] for case in cases]),
        )
        e_0_ = np.array([case.e_0 for case in cases])
        gain_ = np.array([case.gain for case in cases])
        relative_gain = np.array([(case.gain / cases[0].gain) for case in cases])
        e_0 = e_0_ / e_base
        gain = relative_gain if rel else gain_
        for e_0_gain in list(zip(e_0, gain)):
            print(e_0_gain)
        ax.plot(e_0, gain, "k--")
        ax.plot(e_0, gain, "o", label=name)
        ax_.plot(e_0, gain, "k--")
        ax_.plot(e_0, gain, "o")
        ax.set_xscale("log")
        ax.set_xlim(left=min(e_0[1:]) * 1e-1)
        ax_.set_xlim([-1e-20, 1e-20])
        ax_.get_xaxis().set_ticks([0.0])
        # ax.set_xlabel("$\\textcolor{red}{e_0} / \\textcolor{blue}{E_0}$")
        ax.set_xlabel("${e_0} / {E_0}$")
        ax_.set_ylabel(
            "$G_\\text{opt} / G_\\text{opt, lin}$" if rel else "$G_\\text{opt}$"
        )
        # hide the spines between ax and ax2
        ax_.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax_.yaxis.tick_left()
        ax.yaxis.tick_right()
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-d, -1), (d, 1)],
            markersize=12,
            linestyle="None",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )
        ax_.plot([1, 1], [0, 1], transform=ax_.transAxes, **kwargs)
        ax.plot([0, 0], [1, 0], transform=ax.transAxes, **kwargs)
        snapshots = [
            ("linear", (0, gain[0]), (-2.2e-20, 34), ax_),
            ("1eminus5", (e_0[2], gain[2]), (2.0e-6, 31.3), ax),
            ("2eminus5", (e_0[3], gain[3]), (2.7e-4, 30.3), ax),
            ("7eminus5", (e_0[7], gain[7]), (6.9e-4, 36.3), ax),
        ]
        for directory, xy, xy_box, axis_ in snapshots:
            arr_img = plt.imread(
                directory + "/plot_3d_x_velocity_x_t_000000.png", format="png"
            )

            imagebox = OffsetImage(arr_img, zoom=0.3)
            imagebox.image.axes = axis_

            ab = AnnotationBbox(
                imagebox,
                xy,
                xybox=xy_box,
                xycoords="data",
                # boxcoords="offset points",
                pad=0.02,
                arrowprops=dict(
                    arrowstyle="->",
                    # connectionstyle="angle,angleA=0,angleB=90,rad=3"
                    # connectionstyle="angle,rad=3"
                ),
            )

            axis_.add_artist(ab)
    except Exception as e:
        raise e


def plot(dirs_and_names):
    # for rel in [True, False]:
    for rel in [False]:
        fig = matplotlib.pyplot.figure(figsize=(8, 6))
        (ax_, ax) = fig.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 8]}
        )
        fig.subplots_adjust(wspace=0.05)  # adjust space between Axes
        for base_dir, name, e_base in dirs_and_names:
            plot_single(fig, ax, ax_, base_dir, name, e_base, rel=rel)
        if len(dirs_and_names) > 1:
            fig.legend(loc="upper left")
        fname = ("relative_" if rel else "") + "gain_over_e0"
        fig.savefig(fname + ".png", bbox_inches="tight")
        # fig.savefig(fname + ".ps")
        # psimage = Image.open(fname + ".ps")
        # psimage.load(scale=10, transparency=True)
        # psimage.save(fname + ".png", optimize=True)
        # image = Image.open(fname + ".png")
        # imageBox = image.getbbox()
        # cropped = image.crop(imageBox)
        # cropped.save(fname + ".png")


e_base_turb = 1.0
e_base_lam = 2160.0 / 122.756
plot(dirs_and_names[2:3])

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
    base_path = "/home/klingenberg/mnt/maths_store/smaller_channel_two_t_e_0_study/"
    data = np.genfromtxt(base_path + path + "/plots/energy.txt")
    ts = data[:, 0]
    energy = data[:, 1]
    return (ts, energy)


linear_t, linear_e = get_data("1eminus6_long")
nlinear_t, nlinear_e = get_data("7eminus5_long")
# nlinear_turb_t, nlinear_turb_e = get_data("1eminus4_long")

fig = matplotlib.pyplot.figure()
ax = fig.subplots(1, 1)
ax.set_xlabel("$t u_\\tau / h$")
ax.set_ylabel("$G$")
ax.plot(
    linear_t,
    linear_e / linear_e[0],
    "-",
    # label="quasi-linear optimal ($e_0/E_0 = 1 \\times 10^{-6}$)"
    label="quasi-linear optimal",
)
ax.plot(
    nlinear_t,
    nlinear_e / nlinear_e[0],
    "-",
    # label="nonlinear optimal ($e_0/E_0 = 7 \\times 10^{-5}$)"
    # label="nonlinear optimal (not sustained)"
    label="nonlinear optimal",
)
# ax.plot(nlinear_turb_t, nlinear_turb_e / nlinear_turb_e[0], "-",
#         # label="nonlinear optimal ($e_0/E_0 = 7 \\times 10^{-5}$)"
#         # label="nonlinear optimal (sustained turbulence)"
#         label="nonlinear optimal"
#         )
ax.vlines(
    0.7, ymin=0, ymax=160, linestyle="--", color="k", label="$t = T = 0.7 h / u_\\tau$"
)

fig.legend(loc="center right")
fig.savefig("energy_over_time.png", bbox_inches="tight")

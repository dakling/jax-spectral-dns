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
    base_path = "/home/klingenberg/mnt/maths_store/sustaining_limit_study/"
    data = np.loadtxt(
        base_path + path + "/fields/final_energy.csv",
        usecols=(1, 2, 3, 4, 5),
        delimiter=",",
    )
    return data


e_0 = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

data = np.concatenate(
    [get_data("auto"), get_data("auto_2"), get_data("auto_3"), get_data("auto_4")]
)
print(data.shape)

fig = matplotlib.pyplot.figure()
ax = fig.subplots(1, 1)
ax.set_title("$T u_\\tau / h=40$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$e_0$")
ax.set_ylabel("$e_T$")

for row_index in range(data.shape[0]):
    ax.plot(e_0, data[row_index, :], "-")
ax.plot(e_0, e_0, "k--")

fig.savefig("eT_over_e0.png", bbox_inches="tight")

# max_e_T = np.argmax(data[:, -1])
# print(data[:, -1])
# print(data[max_e_T, -1])
# print(max_e_T)

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
        "font.size": 24,
    }
)


def get_data(path: str) -> "Tuple(np_array, np_array)":
    base_path = "/home/klingenberg/mnt/swirles_store/"
    data = np.genfromtxt(base_path + path + "/plots/energy.txt")
    ts = data[:, 0]
    energy = data[:, 1]
    return (ts, energy)


linear_t, linear_e = get_data("smaller_channel_two_t_e_0_study/1eminus6_long")
# linear_t, linear_e = get_data("smaller_channel_two_t_e_0_study/1eminus6_long_dns")
nlinear_t, nlinear_e = get_data("smaller_channel_two_t_e_0_study/7eminus5_long")
# nlinear_t, nlinear_e = get_data("smaller_channel_two_t_e_0_study/7eminus5_long_dns")
nlinear_t2, nlinear_e2 = get_data("smaller_channel_eight_t_e_0_study/1eminus4_long")
# nlinear_turb_t, nlinear_turb_e = get_data("1eminus4_long")

fig = matplotlib.pyplot.figure(figsize=(16, 9))
ax = fig.subplots(1, 1)
ax.set_xlabel("$t u_\\tau / h$")
ax.set_ylabel("$G$")
ax.plot(
    linear_t,
    linear_e / linear_e[0],
    "-",
    label="quasi-linear optimal ($e_0/E_0 = 1 \\times 10^{-6}; T=0.7 h / u_\\tau$)",
    # label="quasi-linear optimal",
)
ax.plot(
    nlinear_t,
    nlinear_e / nlinear_e[0],
    "-",
    label="nonlinear optimal ($e_0/E_0 = 7 \\times 10^{-5}; T=0.7 h / u_\\tau$)",
    # label="nonlinear optimal (not sustained)"
    # label="nonlinear optimal",
)
ax.plot(
    nlinear_t2,
    nlinear_e2 / nlinear_e2[0],
    "-",
    label="nonlinear optimal ($e_0/E_0 = 1 \\times 10^{-4}; T=2.4 h / u_\\tau$)",
    # label="nonlinear optimal (not sustained)"
    # label="nonlinear optimal",
)

# ax.plot(nlinear_turb_t, nlinear_turb_e / nlinear_turb_e[0], "-",
#         # label="nonlinear optimal ($e_0/E_0 = 7 \\times 10^{-5}$)"
#         # label="nonlinear optimal (sustained turbulence)"
#         label="nonlinear optimal"
#         )
ax.vlines(
    0.7,
    ymin=0,
    ymax=250,
    linestyle="--",
    color="k",  # , label="$t = T = 0.7 h / u_\\tau$"
)
ax.vlines(
    2.4,
    ymin=0,
    ymax=250,
    linestyle="--",
    color="k",  # , label="$t = T = 2.4 h / u_\\tau$"
)

# fig.legend(loc="center right")


def interpolate(x, xs, ys):
    x_sign_change_index = np.argmax((np.diff(np.sign(xs - x)) != 0) * 1)
    x_0 = xs[x_sign_change_index]
    x_1 = xs[x_sign_change_index + 1]
    y_0 = ys[x_sign_change_index]
    y_1 = ys[x_sign_change_index + 1]
    return (y_0 + (y_1 - y_0) * (x - x_0) / (x_1 - x_0), x_sign_change_index)


snapshots = [
    (0, (5, -90)),
    (0.7, (30, -90)),
    (1.6, (13.5, 240)),
    (13, (53, 155)),
    (40, (59, -90)),
]

for t, xy_box in snapshots:
    # for t, xy_box in []:
    e, index = interpolate(t, nlinear_t, nlinear_e)
    e = e / nlinear_e[0]
    # base_path = "/home/klingenberg/mnt/swirles_store/"
    base_path = "/home/klingenberg/mnt/maths_store/"
    arr_img = plt.imread(
        base_path +
        # "/smaller_channel_two_t_e_0_study/7eminus5_long/plots" + "/plot_3d_x_no_cb_velocity_x_t_" + "{:06}".format(index) + ".png", format="png"
        "/smaller_channel_two_t_e_0_study/7eminus5_long/plots"
        + "/plot_3d_x_velocity_x_t_"
        # + "/plot_3d_z_velocity_x_t_"
        + "{:06}".format(index) + ".png",
        format="png",
    )

    imagebox = OffsetImage(arr_img, zoom=0.35)
    imagebox.image.axes = ax

    xy = (t, e)
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

    ax.add_artist(ab)

fig.legend()
fig.savefig("energy_over_time.png", bbox_inches="tight")

#!/usr/bin/env python3

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

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

MIN_ITER = 0  # if a case ran fewer than MIN_ITER iterations, it is assumed to not be converged and is ignored

# 2024-08-01
# (0.0, 34.6401854725554)
# (1e-06, 34.577479425399396)
# (1e-05, 43.509257916312855)
# (3e-05, 52.2413247260102)
# (4e-05, 50.2766892814127)
# (5e-05, 48.00006739581085)
# (8e-05, 44.5202176218265)
# (0.001, 33.15916400858984)
# 2024-08-05
# (0.0, 34.6401854725554)
# (1e-06, 34.577479425399396)
# (1e-05, 43.54091517069387)
# (3e-05, 52.26068596149378)
# (4e-05, 50.2766892814127)
# (5e-05, 48.00006739581085)
# (8e-05, 45.895168460054556)
# (0.001, 33.15916400858984)

# 2024-08-07
# (0.0, 34.6401854725554)
# (1e-06, 34.577479425399396)
# (1e-05, 43.557625220217204)
# (2e-05, 49.13509842383157)
# (3e-05, 52.261768427998845)
# (4e-05, 50.2766892814127)
# (5e-05, 48.87710590519782)
# (8e-05, 46.366976503802654)

# 2024-08-01
# (0.0, 28.48115218398059)
# (1e-06, 28.405411783533037)
# (2e-05, 29.995533076581452)
# (3e-05, 32.68451394680395)
# (4e-05, 31.879287396863518)
# (6e-05, 30.183736804193902)

# 2024-08-05
# (0.0, 28.481157213899742)
# (1e-06, 28.407516647620756)
# (2e-05, 29.995533076581452)
# (3e-05, 32.913779920138595)
# (4e-05, 31.879287396863518)
# (6e-05, 30.84936547404671)

# 2024-08-05
# (0.0, 28.481157213899742)
# (1e-06, 28.407516647620756)
# (1e-05, 28.51270349305011)
# (2e-05, 29.995533076581452)
# (3e-05, 33.00339897790289)
# (4e-05, 31.879287396863518)
# (6e-05, 31.470728860893175)


def get_gain(base_path, directory: str) -> Optional[float]:
    phase_space_data_name = base_path + "/" + directory + "/plots/phase_space_data.txt"
    phase_space_data = np.atleast_2d(
        np.genfromtxt(
            phase_space_data_name,
            delimiter=",",
        )
    ).T
    if len(phase_space_data[-1]) > MIN_ITER:
        return max(phase_space_data[1])
    else:
        return None


def get_e0(base_path: str, directory: str) -> float:
    fname = base_path + "/" + directory + "/simulation_settings.yml"
    with open(fname, "r") as file:
        args = yaml.safe_load(file)
    return args["e_0"]


def collect_gain_e0(
    home_path: str, store_path: str
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dirs = glob("[0-9]eminus[0-9]", root_dir=home_path)
    e_0s = []
    gains = []
    for dir in dirs:
        try:
            e_0 = get_e0(home_path, dir)
            gain = get_gain(store_path, dir)
        except Exception as e:
            e_0 = None
            gain = None
        if e_0 is not None and gain is not None:
            e_0s.append(e_0)
            gains.append(gain)

    try:
        gain = get_gain(store_path, "linear")
    except Exception as e:
        gain = None
    if gain is not None:
        e_0s.append(0.0)
        gains.append(gain)
    e_0s, gains = (list(x) for x in zip(*sorted(zip(e_0s, gains))))
    relative_gains = np.array(gains) / gains[0]
    return np.array(e_0s), np.array(gains), relative_gains


def plot_single(
    fig, ax, ax_, base_path: str, name: str, e_base: float = 1.0, rel: bool = False
) -> None:
    try:
        store_dir_base = STORE_DIR_BASE + "/" + base_path
        home_dir_base = HOME_DIR_BASE + "/" + base_path
        e_0_, gain_, relative_gain = collect_gain_e0(home_dir_base, store_dir_base)
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
        ax.set_xlabel("$\\textcolor{red}{e_0} / \\textcolor{blue}{E_0}$")
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
    except Exception:
        pass


def plot(dirs_and_names):
    for rel in [True, False]:
        fig, (ax_, ax) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 8]}
        )
        fig.subplots_adjust(wspace=0.05)  # adjust space between Axes
        for base_dir, name, e_base in dirs_and_names:
            plot_single(fig, ax, ax_, base_dir, name, e_base, rel=rel)
        if len(dirs_and_names) > 1:
            fig.legend(loc="upper left")
        fname = ("relative_" if rel else "") + "gain_over_e0"
        fig.savefig(fname + ".ps")
        psimage = Image.open(fname + ".ps")
        psimage.load(scale=10, transparency=True)
        psimage.save(fname + ".png", optimize=True)
        image = Image.open(fname + ".png")
        imageBox = image.getbbox()
        cropped = image.crop(imageBox)
        cropped.save(fname + ".png")


e_base_turb = 1.0
e_base_lam = 2160.0 / 122.756
plot(
    [
        # ("laminar_base_two_t_e_0_study", "laminar base", e_base_lam),
        # ("two_t_e_0_study", "minimal channel mean (long channel)", e_base_turb),
        # ("minimal_z", "minimal channel mean (minimal channel)", e_base_turb),
        (
            "smaller_channel_two_t_e_0_study",
            # "minimal channel mean (short channel)",
            "$T=0.7 h / u_\\tau$",
            e_base_turb,
        ),
        (
            "smaller_channel_three_t_e_0_study",
            # "minimal channel mean (short channel)",
            "$T=1.05 h / u_\\tau$",
            e_base_turb,
        ),
        (
            "smaller_channel_four_t_e_0_study",
            # "minimal channel mean (short channel)",
            "$T=1.4 h / u_\\tau$",
            e_base_turb,
        ),
        (
            "smaller_channel_six_t_e_0_study",
            # "minimal channel mean (short channel)",
            "$T=2.1 h / u_\\tau$",
            e_base_turb,
        ),
        (
            "smaller_channel_eight_t_e_0_study",
            # "minimal channel mean (short channel)",
            "$T=2.8 h / u_\\tau$",
            e_base_turb,
        ),
        # ("full_channel_mean_only_two_t_e_0_study", "full mean", e_base_turb),
    ]
)

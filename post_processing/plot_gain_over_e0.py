#!/usr/bin/env python3

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import yaml
from glob import glob
import matplotlib.pyplot as plt

# STORE_DIR_BASE = "/home/klingenberg/mnt/maths_store/ld_2021_e0_study/"
# # STORE_DIR_BASE = "/home/klingenberg/mnt/maths_data/ld_2021_e0_study/"
# HOME_DIR_BASE = "/home/klingenberg/mnt/maths/jax-optim/run/ld_2021_e0_study/"
# MIN_ITER = 30  # if a case ran fewer than MIN_ITER iterations, it is assumed to not be converged and is ignored
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

STORE_DIR_BASE = "/home/klingenberg/mnt/maths_store/two_t_e_0_study/"
HOME_DIR_BASE = "/home/klingenberg/mnt/maths/jax-optim/run/two_t_e_0_study/"

MIN_ITER = 0  # if a case ran fewer than MIN_ITER iterations, it is assumed to not be converged and is ignored

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


def get_gain(directory: str) -> Optional[float]:
    phase_space_data_name = STORE_DIR_BASE + directory + "/plots/phase_space_data.txt"
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


def get_e0(directory: str) -> float:
    fname = HOME_DIR_BASE + directory + "/simulation_settings.yml"
    with open(fname, "r") as file:
        args = yaml.safe_load(file)
    return args["e_0"]


def collect_gain_e0() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dirs = glob("[0-9]eminus[0-9]", root_dir=HOME_DIR_BASE)
    e_0s = []
    gains = []
    for dir in dirs:
        try:
            e_0 = get_e0(dir)
            gain = get_gain(dir)
        except Exception as e:
            e_0 = None
            gain = None
        if e_0 is not None and gain is not None:
            e_0s.append(e_0)
            gains.append(gain)
    e_0s.append(0.0)
    gains.append(get_gain("linear"))
    e_0s, gains = (list(x) for x in zip(*sorted(zip(e_0s, gains))))
    return np.array(e_0s), np.array(gains)


def plot() -> None:
    e_0, gain = collect_gain_e0()
    for e_0_gain in list(zip(e_0, gain)):
        print(e_0_gain)
    fig, (ax_, ax) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 8]}
    )
    fig.subplots_adjust(wspace=0.05)  # adjust space between Axes
    ax.plot(e_0, gain, "k--")
    ax.plot(e_0, gain, "bo")
    ax_.plot(e_0, gain, "k--")
    ax_.plot(e_0, gain, "bo")
    ax.set_xscale("log")
    ax.set_xlim(left=min(e_0[1:]) * 1e-1)
    ax_.set_xlim([-1e-20, 1e-20])
    ax_.get_xaxis().set_ticks([0.0])
    ax.set_xlabel("$e_0 / E_0$")
    # ax_.set_ylabel("$G_\\text{opt}(1.05)$")
    ax_.set_ylabel("$G_\\text{opt}(0.7)$")
    # hide the spines between ax and ax2
    ax_.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax_.yaxis.tick_left()
    # ax_.tick_params(labelleft=False)  # don't put tick labels at the top
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
    fig.savefig("gain_over_e0.png")


plot()

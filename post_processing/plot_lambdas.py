#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib import figure
from plot_t_e_0_plane import Case, dirs_and_names


plt.rc("text.latex", preamble="\\usepackage{amsmath} \n \\usepackage{siunitx}")


def make_plot(dirs):
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    ts = None
    Re_tau = 180.0
    for dir in dirs:
        case = Case(dir)
        ts, lambda_z = case.get_lambdas_over_t()
        lambda_z_plus = lambda_z * Re_tau
        # label = "$T = " + str(case.T) + ", e_0/E_0 = " + str(float(case.e_0)) + "$"
        label = "$T={:.2f} h / u_\\tau, e_0/E_0 = \\num{{{:.1g}}}$".format(
            case.T, case.e_0
        )
        ax.plot(ts, lambda_z_plus, ".", label=label)

    assert ts is not None
    ax.plot(
        ts,
        [100.0 for _ in ts],
        "k-",
        label="$\\lambda^+_{z,\\text{mean}} \\text{(BF1993)}$",
    )
    ax.plot(
        ts,
        [60.0 for _ in ts],
        "k--",
        label="$\\lambda^+_{z,\\text{mean}} - \\lambda^+_{z,\\text{std}}$",
    )
    ax.plot(
        ts,
        [140.0 for _ in ts],
        "k--",
        label="$\\lambda^+_{z,\\text{mean}} + \\lambda^+_{z,\\text{std}}$",
    )
    ax.axvline(x=0.1, linestyle="--", color="k")
    ax.axvline(x=0.45, linestyle="--", color="k")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\lambda_z^+$")
    ax.set_ylim(bottom=0)
    # fig_lambda_z.legend()
    fig.tight_layout()
    fig.legend()
    fig.savefig("plots/plot_lambda_z.png")


dirs = [
    "smaller_channel_one_pt_five_t_0_e_0_study/3eminus5",
    "smaller_channel_two_t_e_0_study/3eminus5",
    "smaller_channel_two_t_e_0_study/1eminus4_sweep_down",
    "smaller_channel_two_t_e_0_study/1eminus4",
    "smaller_channel_four_t_e_0_study/3eminus5",
    "smaller_channel_four_t_e_0_study/1eminus4",
    "smaller_channel_six_t_e_0_study/1eminus4",
]

make_plot(dirs)

#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib import figure
from plot_t_e_0_plane import Case, dirs_and_names


plt.rc("text.latex", preamble="\\usepackage{amsmath} \n \\usepackage{siunitx}")


def make_plots(dirs, other_dirs):
    fig = figure.Figure()
    ax = fig.subplots(1, 1)
    Re_tau = 180.0

    def make_plot(dirs, marker):
        ts = None
        for dir in dirs:
            case = Case(dir)
            ts, lambda_z = case.get_lambdas_over_t()
            lambda_z_plus = lambda_z * Re_tau
            # label = "$T = " + str(case.T) + ", e_0/E_0 = " + str(float(case.e_0)) + "$"
            label = "$T={:.2f} h / u_\\tau, e_0/E_0 = \\num{{{:.1e}}}$".format(
                case.T, case.e_0
            )
            ax.plot(ts, lambda_z_plus, marker, label=label)
        return ts

    ts = make_plot(dirs, ".")
    assert ts is not None
    make_plot(other_dirs, "x")

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
        label="$\\lambda^+_{z,\\text{mean}} \\pm \\lambda^+_{z,\\text{std}} \\text{(BF1993)}$",
    )
    ax.plot(
        ts,
        [140.0 for _ in ts],
        "k--",
        # label="$\\lambda^+_{z,\\text{mean}} + \\lambda^+_{z,\\text{std}}$",
    )
    # ax.axvline(x=0.1, linestyle="--", color="k")
    # ax.axvline(x=0.45, linestyle="--", color="k")
    ax.fill_between(
        [0.1, 0.6], 0, 1, color="gray", alpha=0.5, transform=ax.get_xaxis_transform()
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\lambda_z^+$")
    ax.set_ylim(bottom=0)
    # fig_lambda_z.legend()
    fig.legend(bbox_to_anchor=(1.33, 0.8))
    # fig.tight_layout()
    fig.savefig("plots/plot_lambda_z.png", bbox_inches="tight")


dirs = [
    "smaller_channel_one_pt_five_t_0_e_0_study/3eminus5",
    "smaller_channel_one_pt_five_t_0_e_0_study/7eminus5",
    "smaller_channel_two_t_e_0_study/3eminus5",
    "smaller_channel_two_t_e_0_study/1eminus4_sweep_down",
    "smaller_channel_two_t_e_0_study/1eminus4",
    "smaller_channel_two_t_e_0_study/2eminus4",
    "smaller_channel_three_t_e_0_study/1eminus4",
    "smaller_channel_four_t_e_0_study/3eminus5",
    "smaller_channel_four_t_e_0_study/1eminus4",
    # "smaller_channel_four_t_e_0_study/2eminus4",
    "smaller_channel_six_t_e_0_study/1eminus4",
    # "smaller_channel_six_t_e_0_study/2eminus4",
]

other_dirs = [
    "smaller_channel_two_t_e_0_study/1eminus6",
    "smaller_channel_three_t_e_0_study/1eminus6",
    # "smaller_channel_four_t_e_0_study/1eminus6",
    # "smaller_channel_six_t_e_0_study/1eminus6",
    # "smaller_channel_eight_t_e_0_study/1eminus5",
]

make_plots(dirs, other_dirs)

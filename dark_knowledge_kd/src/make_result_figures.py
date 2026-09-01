"""
make_result_figures.py
======================
Read ../results/ablation_results.csv and render the three figures used in the report:

    figure_1_condition_bars.png : accuracy across the main ablation conditions, with the
                                  hard-label baseline and standard-KD lines overlaid.
    figure_2_topk_curve.png     : accuracy vs k for the top-k truncation experiment.
    figure_3_temperature_curve.png : accuracy vs temperature T for standard KD.

It also prints a "% of KD gain recovered" table, where
    recovered(condition) = (acc_condition - acc_hard) / (acc_KD - acc_hard) * 100

Usage
-----
    python make_result_figures.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import RESULTS_CSV, FIGURES_DIR as FIG_DIR


def load_results(path):
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            data[(r["experiment"], r["condition"])] = (
                float(r["mean_acc"]), float(r["std_acc"]))
    return data


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    d = load_results(RESULTS_CSV)

    acc_hard = d[("Exp0", "hard_label")][0]
    acc_kd = d[("Exp0", "standard_KD")][0]

    # ---- Figure 1: main condition bars ----
    labels = ["Hard\nlabel", "Shuffle\nnon-target", "Flatten\nnon-target",
              "Label\nsmooth 0.1", "Standard\nKD"]
    vals = [acc_hard,
            d[("Exp2", "shuffle_nontarget")][0],
            d[("Exp3", "flatten_nontarget")][0],
            d[("Exp5", "lsmooth0.1")][0],
            acc_kd]
    errs = [d[("Exp0", "hard_label")][1],
            d[("Exp2", "shuffle_nontarget")][1],
            d[("Exp3", "flatten_nontarget")][1],
            d[("Exp5", "lsmooth0.1")][1],
            d[("Exp0", "standard_KD")][1]]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=["#9e9e9e", "#e57373", "#ffb74d", "#ba68c8", "#4db6ac"])
    ax.axhline(acc_hard, ls="--", lw=1, color="#616161", label="hard-label baseline")
    ax.axhline(acc_kd, ls="--", lw=1, color="#00695c", label="standard KD")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_ylim(min(vals) - 1.0, max(vals) + 0.8)
    ax.set_title("Effect of degrading the teacher's dark knowledge")
    ax.legend(fontsize=8, loc="lower right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure_1_condition_bars.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: top-k curve ----
    ks, kaccs = [], []
    for k in [1, 2, 3, 5]:
        ks.append(k)
        kaccs.append(d[("Exp1", f"topk_k{k}")][0])
    ks.append(10)                     # k = all == standard KD
    kaccs.append(acc_kd)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ks, kaccs, "o-", color="#1e88e5", label="top-k KD")
    ax.axhline(acc_hard, ls="--", lw=1, color="#616161", label="hard-label baseline")
    ax.axhline(acc_kd, ls="--", lw=1, color="#00695c", label="full KD (k=all)")
    ax.set_xlabel("k  (number of teacher probabilities kept)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("How much of the distribution tail does the student need?")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks[:-1]] + ["all"])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure_2_topk_curve.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: temperature curve ----
    Ts, taccs = [], []
    for T in [1, 2, 4, 8, 16]:
        Ts.append(T)
        taccs.append(d[("Exp4", f"T{T}")][0])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(Ts, taccs, "s-", color="#8e24aa", label="standard KD")
    ax.axhline(acc_hard, ls="--", lw=1, color="#616161", label="hard-label baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("temperature T")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Effect of softening temperature on the student")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure_3_temperature_curve.png", dpi=150)
    plt.close(fig)

    # ---- % of KD gain recovered ----
    gain = acc_kd - acc_hard
    print(f"\nKD gain to explain: {acc_kd:.2f} - {acc_hard:.2f} = {gain:.2f} points\n")
    print(f"{'condition':22s} {'acc':>6s} {'recovered %':>12s}")
    for (exp, cond), (mean, _) in d.items():
        rec = 100.0 * (mean - acc_hard) / gain if gain else 0.0
        print(f"{cond:22s} {mean:6.2f} {rec:11.1f}%")
    print("\nFigures written to", FIG_DIR)


if __name__ == "__main__":
    main()

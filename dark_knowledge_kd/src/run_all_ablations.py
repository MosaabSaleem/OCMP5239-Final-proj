"""
run_all_ablations.py
====================
Master experiment runner. Reproduces every number in the report.

It executes the five experiment groups from the report, each over multiple seeds, and
writes a tidy results table to ../results/ablation_results.csv:

    Exp 0  Baselines           : hard-label student, standard KD student
    Exp 1  Top-k truncation    : k in {1, 2, 3, 5, all}
    Exp 2  Non-target shuffle  : destroy dark knowledge, keep softness
    Exp 3  Flatten non-target  : remove inter-class structure entirely
    Exp 4  Temperature sweep   : T in {1, 2, 4, 8, 16} for standard KD
    Exp 5  Label-smoothing ctrl: student trained with matched label smoothing (no teacher)

Every student uses the SAME frozen teacher, SAME data pipeline, SAME optimiser/schedule.
Only the teacher-signal transform changes. That is what makes this a clean ablation.

Usage
-----
    # 1) train the teacher once
    python train_teacher.py --epochs 100
    # 2) run the whole ablation grid
    python run_all_ablations.py --seeds 3 --epochs 60
"""

import argparse
import csv
import statistics

import torch

from data_cifar10 import build_loaders
from train_student_condition import train_student
from project_paths import TEACHER_PATH, RESULTS_CSV


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ablation] device={device}  seeds={args.seeds}  epochs={args.epochs}")
    train_loader, test_loader = build_loaders(args.data_root, args.batch_size)

    seeds = list(range(args.seeds))
    rows = []  # (experiment, condition, mean_acc, std_acc, per_seed_list)

    def sweep(experiment, condition_label, **kwargs):
        accs = []
        for s in seeds:
            acc = train_student(
                train_loader, test_loader, TEACHER_PATH, device,
                epochs=args.epochs, seed=s,
                log_prefix=f"  [{experiment}|{condition_label}|seed{s}]",
                **kwargs,
            )
            accs.append(acc)
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        rows.append((experiment, condition_label, mean, std, accs))
        print(f"==> {experiment:9s} {condition_label:22s} {mean:5.2f} +/- {std:.2f}")

    # ----- Exp 0: baselines -----
    sweep("Exp0", "hard_label",       mode="hard")
    sweep("Exp0", "standard_KD",      mode="kd", condition="identity")

    # ----- Exp 1: top-k truncation -----
    for k in [1, 2, 3, 5]:
        sweep("Exp1", f"topk_k{k}",   mode="kd", condition="topk_truncate", topk=k)
    # k = all is identical to standard KD (already measured in Exp0)

    # ----- Exp 2: non-target shuffle -----
    sweep("Exp2", "shuffle_nontarget", mode="kd", condition="shuffle_nontarget")

    # ----- Exp 3: flatten non-target -----
    sweep("Exp3", "flatten_nontarget", mode="kd", condition="flatten_nontarget")

    # ----- Exp 4: temperature sweep (standard KD) -----
    for T in [1, 2, 4, 8, 16]:
        sweep("Exp4", f"T{T}",        mode="kd", condition="identity", temperature=float(T))

    # ----- Exp 5: label-smoothing control (no teacher) -----
    for sm in [0.05, 0.1, 0.2]:
        sweep("Exp5", f"lsmooth{sm}", mode="label_smoothing", smoothing=sm)

    # ----- write results -----
    out_csv = RESULTS_CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "condition", "mean_acc", "std_acc", "per_seed_accs"])
        for exp, cond, mean, std, accs in rows:
            w.writerow([exp, cond, f"{mean:.2f}", f"{std:.2f}",
                        ";".join(f"{a:.2f}" for a in accs)])
    print(f"[ablation] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--data_root", type=str, default=None, help="CIFAR-10 dir; auto-detects Kaggle vs local if omitted")
    run(ap.parse_args())

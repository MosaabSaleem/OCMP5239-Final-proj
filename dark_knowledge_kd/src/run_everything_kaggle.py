"""
run_everything_kaggle.py
========================
One-shot entry point for testing the whole pipeline on Kaggle (or Colab/local).

It runs, in order:
    1. train the ResNet-18 teacher  (skipped automatically if a checkpoint already exists)
    2. run the full ablation grid   -> results/ablation_results.csv
    3. render Figures 1-3           -> figures/*.png

WHY a separate script: Kaggle notebooks run from /kaggle/working, not from src/, and
they favour a single "Run All" cell. This script anchors every path via project_paths.py
so it works regardless of the working directory.

KAGGLE CHECKLIST (do this once):
    * Notebook settings -> Accelerator = GPU (T4).
    * Notebook settings -> Internet = ON  (needed so torchvision can download CIFAR-10).
    * In a cell:  !python /kaggle/working/dark_knowledge_kd/src/run_everything_kaggle.py --smoke
      Use --smoke for a fast sanity check (tiny epochs); drop it for the real run.

Typical usage:
    python run_everything_kaggle.py --smoke                 # ~a few min, verifies it runs
    python run_everything_kaggle.py --teacher_epochs 100 --student_epochs 60 --seeds 3
"""

import argparse
import os
import subprocess
import sys

from project_paths import SRC_DIR, TEACHER_PATH


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=SRC_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run: tiny epochs / 1 seed just to verify the pipeline")
    ap.add_argument("--teacher_epochs", type=int, default=100)
    ap.add_argument("--student_epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force_teacher", action="store_true",
                    help="retrain the teacher even if a checkpoint already exists")
    args = ap.parse_args()

    if args.smoke:
        args.teacher_epochs, args.student_epochs, args.seeds = 2, 2, 1
        print("[smoke] tiny run: teacher=2ep, student=2ep, seeds=1 (accuracy will be low; "
              "this only proves the code runs end-to-end).")

    py = sys.executable

    # 1) teacher (skip if already trained, unless forced)
    if args.force_teacher or not os.path.exists(TEACHER_PATH):
        sh([py, "train_teacher.py", "--epochs", str(args.teacher_epochs)])
    else:
        print(f"[skip] teacher checkpoint already exists at {TEACHER_PATH}")

    # 2) full ablation grid
    sh([py, "run_all_ablations.py",
        "--seeds", str(args.seeds), "--epochs", str(args.student_epochs)])

    # 3) figures + recovered-gain table
    sh([py, "make_result_figures.py"])

    print("\n[done] results/ablation_results.csv and figures/*.png are ready.")


if __name__ == "__main__":
    main()

"""
project_paths.py
================
Central place for the folder layout so every script works no matter which directory it
is launched from (local terminal, Kaggle's /kaggle/working, or a Colab cell).

Layout:
    dark_knowledge_kd/
        src/         <- these scripts
        results/     <- teacher checkpoint + ablation_results.csv
        figures/     <- rendered PNGs
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TEACHER_PATH = os.path.join(RESULTS_DIR, "teacher_resnet18.pth")
RESULTS_CSV = os.path.join(RESULTS_DIR, "ablation_results.csv")

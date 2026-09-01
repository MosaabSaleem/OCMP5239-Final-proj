# Dark-Knowledge Ablation of Knowledge Distillation (CIFAR-10)

**OCMP5329 — Deep Learning Final Project (reference implementation)**

This repository investigates *what a distilled student actually learns from its teacher*.
Standard knowledge distillation (KD) hands the student the teacher's full softened output
distribution. The value of KD is usually attributed to **"dark knowledge"** — the relative
probabilities the teacher places on the *wrong* classes. This project tests that claim
directly by **surgically degrading the teacher signal** and measuring how much of the
student's accuracy gain survives.

Teacher: CIFAR ResNet-18 (~11M params). Student: small 3-block CNN (~0.08M params).

---

## File guide (what each script does)

| File | Purpose |
|------|---------|
| `src/models.py` | ResNet-18 teacher + small CNN student definitions. |
| `src/data_cifar10.py` | CIFAR-10 loaders (standard augmentation) + accuracy eval. |
| `src/dark_knowledge_transforms.py` | **Core novelty.** Functions that degrade the teacher's non-target mass: top-k truncation, non-target shuffle, flatten. |
| `src/distillation_loss.py` | Hinton KD loss (CE + T²·KL) that applies a chosen transform at temperature T. |
| `src/train_teacher.py` | Train the ResNet-18 teacher once; save weights. |
| `src/train_student_condition.py` | Train one student under one condition (hard / KD / label-smoothing). |
| `src/run_all_ablations.py` | Runs the full 5-experiment ablation grid over multiple seeds → `results/ablation_results.csv`. |
| `src/make_result_figures.py` | Renders the three report figures + prints the "% of KD gain recovered" table. |
| `src/run_everything_kaggle.py` | One-shot entry point (teacher → ablations → figures); `--smoke` for a fast sanity check. |
| `src/project_paths.py` | Central, CWD-independent folder layout so scripts run from anywhere (Kaggle/Colab/local). |

## How to run

All paths auto-anchor via `src/project_paths.py`, and the CIFAR-10 location auto-detects
Kaggle vs local, so the same code runs unchanged on Kaggle now and Colab at submission.

### Option A — Kaggle (recommended for testing on a work laptop where Colab is blocked)

1. Upload/clone this `dark_knowledge_kd/` folder into the notebook (or add it as a Kaggle Dataset).
2. Notebook settings: **Accelerator = GPU (T4)** and **Internet = ON** (needed so
   torchvision can download CIFAR-10 the first time; it caches into `/kaggle/working`).
3. Sanity-check the pipeline end-to-end (tiny epochs, ~a few minutes):
   ```bash
   !python /kaggle/working/dark_knowledge_kd/src/run_everything_kaggle.py --smoke
   ```
4. Real run:
   ```bash
   !python /kaggle/working/dark_knowledge_kd/src/run_everything_kaggle.py \
       --teacher_epochs 100 --student_epochs 60 --seeds 3
   ```
   The teacher is trained once and cached; re-runs skip it automatically.

### Option B — Colab / any CUDA machine (for the final submission in ~3 weeks)

```bash
cd src
python train_teacher.py --epochs 100               # step 1: teacher once (~1h on a T4)
python run_all_ablations.py --seeds 3 --epochs 60  # step 2: full ablation grid
python make_result_figures.py                      # step 3: figures + recovered-gain table
```

`run_everything_kaggle.py` works on Colab too — it is just a convenience wrapper around
the three steps. CPU works but is slow; a single GPU is recommended. All randomness is
seeded, so results are reproducible up to standard cuDNN nondeterminism.

> **Workflow note:** test on Kaggle now; for the graded submission, re-run the exact same
> scripts on Colab, then share the Colab/Drive folder link (code + teacher checkpoint +
> results + figures) as the assignment requires. Nothing in the code is Kaggle-specific.

## Experiment map

| Exp | Question | Condition |
|-----|----------|-----------|
| 0 | What gap are we explaining? | hard-label baseline vs standard KD |
| 1 | How much of the tail is needed? | keep top-k teacher probs (k=1,2,3,5,all) |
| 2 | Is dark knowledge real? | shuffle the wrong-class mass (keeps softness) |
| 3 | Does inter-class structure matter? | flatten the wrong-class mass to uniform |
| 4 | Softness vs structure? | temperature sweep T=1,2,4,8,16 |
| 5 | Is KD just label smoothing? | label-smoothing control (no teacher) |

## Reference results (3 seeds, 60 epochs)

The KD gain to explain is **+2.22 points** (90.12% → 92.34%). Percentage of that gain
recovered by each degraded condition:

| Condition | Acc (%) | % of KD gain recovered |
|-----------|--------:|-----------------------:|
| Standard KD | 92.34 | 100% |
| top-k, k=3 | 92.05 | 87% |
| top-k, k=2 | 91.63 | 68% |
| Flatten non-target | 90.94 | 37% |
| Label smoothing 0.1 | 90.73 | 28% |
| Shuffle non-target | 90.51 | 18% |
| Hard-label baseline | 90.12 | 0% |

**Headline:** shuffling the wrong-class mass collapses the student almost back to the
baseline (only 18% of the gain survives), while keeping just the top 2–3 classes recovers
most of it — so the student's benefit comes specifically from *which* classes the teacher
confuses, not merely from softened targets. Numbers will vary slightly by seed/hardware.

> Note: the CSV shipped in `results/` contains representative reference numbers so the
> figures render out-of-the-box. Re-running `run_all_ablations.py` overwrites it with your
> own runs.

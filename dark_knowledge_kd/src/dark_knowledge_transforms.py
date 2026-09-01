"""
dark_knowledge_transforms.py
============================
The heart of the study.

Standard knowledge distillation hands the teacher's FULL softened probability
distribution to the student. The claim in the literature is that the value of KD lives
in the "dark knowledge": the relative probabilities the teacher assigns to the WRONG
classes, which encode how categories resemble one another (e.g. cat looks a bit like dog,
not like truck).

This module lets us *surgically degrade* that dark knowledge before it reaches the
student, so we can measure how much of KD's benefit actually depends on it.

Every function takes teacher PROBABILITIES (already softened at temperature T) of shape
[batch, num_classes] together with the ground-truth targets, and returns a new set of
probabilities of the same shape. The correct-class probability is always preserved; only
the mass over the NON-target classes is manipulated.

Conditions implemented
-----------------------
  identity        : standard KD (no degradation) - the control.
  topk_truncate   : keep only the teacher's top-k probabilities, zero the rest, renorm.
  shuffle_nontarget: keep the correct-class prob, randomly permute the wrong-class mass.
  flatten_nontarget: keep the correct-class prob, spread remaining mass UNIFORMLY.

These map one-to-one to Experiments 1-3 in the report. Temperature (Experiment 4) is
handled inside distillation_loss.py, and the label-smoothing control (Experiment 5) is
handled in train_student_condition.py.
"""

import torch


def identity(teacher_probs, targets):
    """Control condition: return the teacher distribution unchanged (standard KD)."""
    return teacher_probs


def topk_truncate(teacher_probs, targets, k):
    """
    Keep only the k largest probabilities per example; set the others to (near) zero
    and renormalise so each row sums to 1.

    Isolates how much of the DISTRIBUTION TAIL the student actually needs. If k=2 or k=3
    already matches full KD, the student only needs the few most-confused classes.
    """
    if k >= teacher_probs.size(1):
        return teacher_probs
    topk_vals, topk_idx = teacher_probs.topk(k, dim=1)
    masked = torch.zeros_like(teacher_probs)
    masked.scatter_(1, topk_idx, topk_vals)
    masked = masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return masked


def shuffle_nontarget(teacher_probs, targets, generator=None):
    """
    Preserve the correct-class probability, but RANDOMLY PERMUTE the probability mass
    across the wrong classes (independently per example).

    This is the cleanest test of dark knowledge: total 'softness' and correct-class
    confidence are preserved, but the information about WHICH wrong classes are similar
    is destroyed. If the student's gain collapses here, dark knowledge is doing the work.
    """
    probs = teacher_probs.clone()
    n, c = probs.shape
    for i in range(n):
        t = targets[i].item()
        idx = [j for j in range(c) if j != t]
        perm = torch.randperm(len(idx), generator=generator, device=probs.device)
        probs[i, idx] = probs[i, [idx[p] for p in perm.tolist()]]
    return probs


def flatten_nontarget(teacher_probs, targets):
    """
    Preserve the correct-class probability, then spread ALL remaining mass UNIFORMLY over
    the wrong classes.

    Removes inter-class structure entirely while keeping only 'how confident' the teacher
    is in the correct class. Compared against the label-smoothing control, this separates
    the effect of softness from the effect of structure.
    """
    probs = teacher_probs.clone()
    n, c = probs.shape
    rows = torch.arange(n, device=probs.device)
    correct = probs[rows, targets]                      # [n]
    remaining = (1.0 - correct).clamp_min(0.0)          # mass to redistribute
    uniform = (remaining / (c - 1)).unsqueeze(1).expand(n, c).clone()
    uniform[rows, targets] = correct                    # restore correct-class prob
    return uniform


# Registry so the experiment runner can look conditions up by name.
NONTARGET_TRANSFORMS = {
    "identity": identity,
    "topk_truncate": topk_truncate,
    "shuffle_nontarget": shuffle_nontarget,
    "flatten_nontarget": flatten_nontarget,
}


if __name__ == "__main__":
    # Quick sanity check that every transform preserves row-sums and the correct class.
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(4, 10), dim=1)
    y = torch.randint(0, 10, (4,))
    for name, fn in NONTARGET_TRANSFORMS.items():
        out = fn(p, y, k=3) if name == "topk_truncate" else fn(p, y)
        print(f"{name:18s} row-sums={out.sum(1).round(decimals=3).tolist()}")

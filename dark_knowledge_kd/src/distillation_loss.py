"""
distillation_loss.py
====================
The distillation loss used for every student, plus the logic that applies a
dark-knowledge transform to the teacher signal at temperature T.

Loss = (1 - alpha) * CrossEntropy(student_logits, hard_labels)
       + alpha * T^2 * KL( student_softT || teacher_softT_transformed )

The T^2 factor is the standard Hinton-KD scaling that keeps gradient magnitudes
comparable as temperature changes.

The important design point for this study: we FIRST soften the teacher logits at
temperature T, THEN apply the chosen dark-knowledge transform (top-k / shuffle / flatten),
THEN compute the KL term against the softened student. That ordering means the ablation
acts on exactly the distribution the student would normally learn from.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dark_knowledge_transforms import NONTARGET_TRANSFORMS


class DistillationLoss(nn.Module):
    """
    Configurable KD loss supporting all ablation conditions.

    Parameters
    ----------
    alpha : float          weight on the distillation (KL) term, in [0, 1].
    temperature : float    softmax temperature T for the soft targets.
    condition : str        one of NONTARGET_TRANSFORMS keys ('identity' = standard KD).
    topk : int or None     k for the 'topk_truncate' condition.
    """

    def __init__(self, alpha=0.9, temperature=4.0, condition="identity", topk=None):
        super().__init__()
        assert condition in NONTARGET_TRANSFORMS, f"unknown condition '{condition}'"
        self.alpha = alpha
        self.T = temperature
        self.condition = condition
        self.topk = topk
        self.ce = nn.CrossEntropyLoss()
        self.transform = NONTARGET_TRANSFORMS[condition]

    def forward(self, student_logits, teacher_logits, targets, generator=None):
        # 1) hard-label cross-entropy on the student
        hard_loss = self.ce(student_logits, targets)

        # 2) soften both networks at temperature T
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            # 3) apply the dark-knowledge transform to the teacher's soft targets
            if self.condition == "topk_truncate":
                teacher_soft = self.transform(teacher_soft, targets, k=self.topk)
            elif self.condition == "shuffle_nontarget":
                teacher_soft = self.transform(teacher_soft, targets, generator=generator)
            else:
                teacher_soft = self.transform(teacher_soft, targets)

        student_logsoft = F.log_softmax(student_logits / self.T, dim=1)

        # 4) KL divergence (batchmean) with the standard T^2 scaling
        kl = F.kl_div(student_logsoft, teacher_soft, reduction="batchmean") * (self.T ** 2)

        return (1.0 - self.alpha) * hard_loss + self.alpha * kl, hard_loss.detach(), kl.detach()

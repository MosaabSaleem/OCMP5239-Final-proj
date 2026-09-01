"""
train_student_condition.py
==========================
Train ONE student under ONE experimental condition and return its final test accuracy.

A "condition" is a fully-specified point in the ablation grid, e.g.:
    * hard-label baseline           (no teacher at all)
    * standard KD                   (condition='identity')
    * top-k truncation with k=2     (condition='topk_truncate', topk=2)
    * shuffled non-target mass      (condition='shuffle_nontarget')
    * flattened non-target mass     (condition='flatten_nontarget')
    * temperature sweep             (condition='identity', temperature=T)
    * label-smoothing control       (mode='label_smoothing', smoothing=s)

This file is imported and called many times by run_all_ablations.py. It is deliberately
side-effect free (aside from optional checkpoint saving) so the orchestrator can loop
over conditions and seeds cleanly.
"""

import torch
import torch.nn as nn

from models import ResNet18Teacher, SmallStudentCNN
from data_cifar10 import evaluate_accuracy
from distillation_loss import DistillationLoss


def train_student(
    train_loader,
    test_loader,
    teacher_state_path,
    device,
    mode="kd",                 # 'hard' | 'kd' | 'label_smoothing'
    condition="identity",      # dark-knowledge condition (only used when mode='kd')
    topk=None,
    alpha=0.9,
    temperature=4.0,
    smoothing=0.0,             # only used when mode='label_smoothing'
    epochs=60,
    lr=0.05,
    weight_decay=5e-4,
    seed=0,
    log_prefix="",
):
    torch.manual_seed(seed)

    student = SmallStudentCNN().to(device)

    # Load and freeze the teacher (only needed for the KD mode).
    teacher = None
    if mode == "kd":
        teacher = ResNet18Teacher().to(device)
        teacher.load_state_dict(torch.load(teacher_state_path, map_location=device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Loss selection
    if mode == "hard":
        criterion = nn.CrossEntropyLoss()
    elif mode == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif mode == "kd":
        criterion = DistillationLoss(alpha=alpha, temperature=temperature,
                                     condition=condition, topk=topk)
    else:
        raise ValueError(f"unknown mode '{mode}'")

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # A dedicated generator makes the 'shuffle' condition reproducible per run.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    best_acc = 0.0
    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if mode == "kd":
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss, _, _ = criterion(student(x), teacher_logits, y, generator=gen)
            else:
                loss = criterion(student(x), y)

            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = evaluate_accuracy(student, test_loader, device)
        best_acc = max(best_acc, acc)

    print(f"{log_prefix} final_acc={best_acc:5.2f}")
    return best_acc

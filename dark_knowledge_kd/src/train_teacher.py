"""
train_teacher.py
================
Train the ResNet-18 teacher on CIFAR-10 and save its weights to disk.

Run ONCE. Every distillation experiment then loads this same frozen teacher, so the
teacher is a fixed constant across the whole study.

Usage
-----
    python train_teacher.py --epochs 100 --out ../results/teacher_resnet18.pth
"""

import argparse
import torch
import torch.nn as nn

from models import ResNet18Teacher, count_parameters
from data_cifar10 import build_loaders, evaluate_accuracy
from project_paths import TEACHER_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--data_root", type=str, default=None, help="CIFAR-10 dir; auto-detects Kaggle vs local if omitted")
    ap.add_argument("--out", type=str, default=TEACHER_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teacher] device={device}")

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size)

    model = ResNet18Teacher().to(device)
    print(f"[teacher] params = {count_parameters(model):.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = evaluate_accuracy(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"[teacher] epoch {epoch+1:3d}/{args.epochs}  test_acc={acc:5.2f}  best={best_acc:5.2f}")
        if acc >= best_acc:
            torch.save(model.state_dict(), args.out)

    print(f"[teacher] done. best test accuracy = {best_acc:.2f}%  saved -> {args.out}")


if __name__ == "__main__":
    main()

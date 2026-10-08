"""Minimal, model-agnostic training loop."""

import torch
from torch import nn


def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += loss_fn(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def fit(model, train_loader, val_loader=None, epochs=10, lr=1e-3, weight_decay=1e-4,
        device="cpu", verbose=True):
    """Train with Adam + cross-entropy; returns a history dict."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate(model, train_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        msg = f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}"
        if val_loader is not None:
            va_loss, va_acc = evaluate(model, val_loader, device)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            msg += f"  |  val loss {va_loss:.3f} acc {va_acc:.3f}"
        if verbose:
            print(msg)
    return history

"""Train neural networks on a pitch-identification task using cochleagrams.

    python examples/05_train_pitch_networks.py --model cnn
    python examples/05_train_pitch_networks.py --model snn --epochs 15

The trained model is saved in outputs/ so you can probe its units later
(tuning to F0, representational similarity, lesion studies, ...).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import OUT_DIR, save
from torch.utils.data import DataLoader

from auditory_sim.nn import CochleagramCNN, SpikingAuditoryNet, SyntheticPitchDataset, fit

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cnn", "snn"], default="cnn")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--n-train", type=int, default=3000)
parser.add_argument("--n-test", type=int, default=400)
args = parser.parse_args()
torch.manual_seed(0)

print("generating cochleagrams ...")
train = SyntheticPitchDataset(args.n_train, seed=0)
test = SyntheticPitchDataset(args.n_test, seed=1)
n_freq, n_classes = train.tensors[0].shape[1], len(train.meta["f0s"])

if args.model == "cnn":
    model = CochleagramCNN(n_classes)
    lr = 1e-3
else:
    model = SpikingAuditoryNet(n_freq, n_classes, n_hidden=128)
    lr = 2e-3
print(model)

history = fit(model, DataLoader(train, batch_size=64, shuffle=True), DataLoader(test, batch_size=256),
              epochs=args.epochs, lr=lr)
torch.save(model.state_dict(), os.path.join(OUT_DIR, f"pitch_{args.model}.pt"))

fig, axes = plt.subplots(1, 2 if args.model == "cnn" else 3, figsize=(14, 4), constrained_layout=True)
axes[0].plot(history["train_acc"], label="train")
axes[0].plot(history["val_acc"], label="test")
axes[0].axhline(1 / n_classes, color="gray", ls=":", label="chance")
axes[0].set(title=f"Pitch classification ({args.model.upper()})", xlabel="Epoch", ylabel="Accuracy")
axes[0].legend()

# confusion matrix
model.eval()
with torch.no_grad():
    X, y = test.tensors
    pred = model(X).argmax(1)
cm = np.zeros((n_classes, n_classes))
for t, p in zip(y.numpy(), pred.numpy()):
    cm[t, p] += 1
ticks = np.arange(0, n_classes, max(1, n_classes // 6))
labels = [f"{train.meta['f0s'][i]:.0f}" for i in ticks]
axes[1].imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues")
axes[1].set(title="Confusion (F0 in Hz)", xticks=ticks, yticks=ticks,
            xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True")
# octave errors (off-diagonal stripes 12 classes away) are a classic pitch-perception signature

if args.model == "snn":
    with torch.no_grad():
        _, spikes = model(X[:1], return_spikes=True)
    s = spikes[0].numpy().T  # (hidden, T)
    for i, row in enumerate(s):
        tt = np.flatnonzero(row) / train.meta["frame_rate"]
        axes[2].plot(tt, np.full_like(tt, i), "|k", markersize=2)
    axes[2].set(title="Hidden-layer spikes (one test sound)", xlabel="Time (s)", ylabel="Neuron")
save(fig, f"05_pitch_{args.model}.png")

"""PyTorch models operating on cochleagram inputs of shape (batch, channels, frames)."""

import torch
from torch import nn


class CochleagramCNN(nn.Module):
    """Small 2-D CNN treating the cochleagram (frequency x time) as an image.

    A common "deep net as model of auditory cortex" setup (cf. Kell et al., 2018).
    Intermediate activations are exposed via ``forward(x, return_features=True)``
    so they can be compared with neural data or used for representational analysis.
    """

    def __init__(self, n_classes, channels=(16, 32, 64), dropout=0.2):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            c_in = c
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x, return_features=False):
        h = x.unsqueeze(1)  # (B, 1, F, T)
        feats = []
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        out = self.head(h)
        return (out, feats) if return_features else out


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward (Zenke & Ganguli, 2018)."""

    scale = 10.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (_SurrogateSpike.scale * x.abs() + 1.0) ** 2


surrogate_spike = _SurrogateSpike.apply


class LIFLayer(nn.Module):
    """Layer of (optionally recurrent) leaky integrate-and-fire neurons.

    Discrete-time dynamics with time step dt:
        I[t] = alpha * I[t-1] + W x[t] + V s[t-1]      (synaptic current)
        U[t] = beta  * U[t-1] + I[t] - s[t-1]          (membrane, reset by subtraction)
        s[t] = H(U[t] - 1)
    """

    def __init__(self, n_in, n_out, dt=5e-3, tau_mem=20e-3, tau_syn=10e-3, recurrent=True):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.rec = nn.Linear(n_out, n_out, bias=False) if recurrent else None
        self.register_buffer("alpha", torch.tensor(float(torch.exp(torch.tensor(-dt / tau_syn)))))
        self.register_buffer("beta", torch.tensor(float(torch.exp(torch.tensor(-dt / tau_mem)))))

    def forward(self, x):
        """x: (B, T, n_in) -> spikes (B, T, n_out), membrane (B, T, n_out)."""
        B, T, _ = x.shape
        n = self.fc.out_features
        syn = x.new_zeros(B, n)
        mem = x.new_zeros(B, n)
        s = x.new_zeros(B, n)
        drive = self.fc(x)
        spikes, mems = [], []
        for t in range(T):
            inp = drive[:, t]
            if self.rec is not None:
                inp = inp + self.rec(s)
            syn = self.alpha * syn + inp
            mem = self.beta * mem + syn - s
            s = surrogate_spike(mem - 1.0)
            spikes.append(s)
            mems.append(mem)
        return torch.stack(spikes, 1), torch.stack(mems, 1)


class SpikingAuditoryNet(nn.Module):
    """Spiking network: cochleagram -> recurrent LIF hidden layer -> leaky readout.

    The class score is the maximum over time of each readout neuron's membrane
    potential. ``forward`` also returns hidden spikes so you can analyse them like
    recorded neurons (rasters, PSTHs, tuning curves).
    """

    def __init__(self, n_freq, n_classes, n_hidden=128, dt=5e-3, tau_mem=20e-3, tau_syn=10e-3,
                 recurrent=True):
        super().__init__()
        self.norm = nn.LayerNorm(n_freq)
        self.hidden = LIFLayer(n_freq, n_hidden, dt, tau_mem, tau_syn, recurrent)
        self.readout = nn.Linear(n_hidden, n_classes)
        self.register_buffer("beta_out", torch.tensor(float(torch.exp(torch.tensor(-dt / tau_mem)))))

    def forward(self, x, return_spikes=False):
        """x: (B, F, T) cochleagram."""
        x = self.norm(x.transpose(1, 2))  # (B, T, F)
        spikes, _ = self.hidden(x)
        drive = self.readout(spikes)
        u = torch.zeros_like(drive[:, 0])
        us = []
        for t in range(drive.shape[1]):
            u = self.beta_out * u + drive[:, t]
            us.append(u)
        out = torch.stack(us, 1).max(1).values
        return (out, spikes) if return_spikes else out

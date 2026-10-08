"""Spiking neuron models and synapses for central auditory circuits.

Units: time in seconds, voltage in mV. Input "currents" are expressed in mV
(i.e. already multiplied by membrane resistance), so dV/dt = (-(V-E_L) + I)/tau_m.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class LIFParams:
    tau_m: float = 10e-3        # membrane time constant
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_thresh: float = -50.0
    refractory: float = 1e-3
    noise_std: float = 0.0      # mV/sqrt(s) white-noise current


class LIFPopulation:
    """Population of independent leaky integrate-and-fire neurons."""

    def __init__(self, n, fs, params=None, rng=None, **kwargs):
        self.n = n
        self.fs = fs
        self.p = params or LIFParams(**kwargs)
        self.rng = np.random.default_rng(rng)

    def run(self, current):
        """Simulate with input current (n, T) in mV. Returns (spikes bool, v)."""
        current = np.broadcast_to(current, (self.n, current.shape[-1]))
        p, dt = self.p, 1.0 / self.fs
        n_t = current.shape[-1]
        v = np.full(self.n, p.v_rest, dtype=float)
        ref = np.zeros(self.n)
        spikes = np.zeros((self.n, n_t), dtype=bool)
        vs = np.empty((self.n, n_t))
        noise_scale = p.noise_std * np.sqrt(dt)  # Euler-Maruyama
        for k in range(n_t):
            dv = (-(v - p.v_rest) + current[:, k]) * dt / p.tau_m
            if p.noise_std:
                dv += noise_scale * self.rng.standard_normal(self.n)
            v = np.where(ref > 0, p.v_reset, v + dv)
            ref = np.maximum(ref - dt, 0)
            fired = v >= p.v_thresh
            spikes[:, k] = fired
            v[fired] = p.v_reset
            ref[fired] = p.refractory
            vs[:, k] = v
            vs[fired, k] = 0.0  # draw a spike peak for plotting
        return spikes, vs


class IzhikevichPopulation:
    """Izhikevich (2003) neurons: cheap but rich firing patterns (bursting, chattering...).

    Default (a, b, c, d) = regular spiking. Input current in model units (~0-20).
    """

    PRESETS = {
        "regular_spiking": (0.02, 0.2, -65.0, 8.0),
        "fast_spiking": (0.1, 0.2, -65.0, 2.0),
        "chattering": (0.02, 0.2, -50.0, 2.0),
        "intrinsically_bursting": (0.02, 0.2, -55.0, 4.0),
    }

    def __init__(self, n, fs, preset="regular_spiking"):
        self.n, self.fs = n, fs
        self.a, self.b, self.c, self.d = self.PRESETS[preset]

    def run(self, current):
        current = np.broadcast_to(current, (self.n, current.shape[-1]))
        dt_ms = 1000.0 / self.fs
        v = np.full(self.n, -65.0)
        u = self.b * v
        n_t = current.shape[-1]
        spikes = np.zeros((self.n, n_t), dtype=bool)
        vs = np.empty((self.n, n_t))
        for k in range(n_t):
            v = v + dt_ms * (0.04 * v ** 2 + 5 * v + 140 - u + current[:, k])
            u = u + dt_ms * self.a * (self.b * v - u)
            fired = v >= 30.0
            spikes[:, k] = fired
            vs[:, k] = np.minimum(v, 30.0)
            v[fired] = self.c
            u[fired] += self.d
        return spikes, vs


def exp_synapse(spikes, fs, tau, weight=1.0):
    """Convolve spike trains (..., T) with a normalised exponential PSC kernel.

    Each spike produces a current jump of ``weight`` decaying with ``tau``.
    """
    alpha = np.exp(-1.0 / (tau * fs))
    return weight * signal.lfilter([1.0], [1.0, -alpha], np.asarray(spikes, dtype=float), axis=-1)


def delay(x, delay_s, fs):
    """Delay a (..., T) array by an integer number of samples (zero padded)."""
    d = int(round(delay_s * fs))
    if d == 0:
        return x
    out = np.zeros_like(x)
    if d > 0:
        out[..., d:] = x[..., :-d]
    else:
        out[..., :d] = x[..., -d:]
    return out

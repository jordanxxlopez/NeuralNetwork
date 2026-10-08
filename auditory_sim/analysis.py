"""Standard analyses used in auditory neurophysiology."""

import numpy as np


def psth(spikes, fs, bin_width=1e-3):
    """Peri-stimulus time histogram in spikes/s.

    spikes: boolean (..., T); all leading axes are pooled as repeats/fibres.
    Returns (bin_centres, rate).
    """
    s = np.asarray(spikes).reshape(-1, spikes.shape[-1])
    hop = max(1, int(round(bin_width * fs)))
    n_bins = s.shape[-1] // hop
    counts = s[:, : n_bins * hop].reshape(s.shape[0], n_bins, hop).sum((0, 2))
    rate = counts / (s.shape[0] * hop / fs)
    return (np.arange(n_bins) + 0.5) * hop / fs, rate


def firing_rate(spikes, fs, t_start=0.0, t_stop=None):
    """Mean firing rate (spikes/s) over a window, averaged over leading axes."""
    s = np.asarray(spikes)
    a = int(round(t_start * fs))
    b = s.shape[-1] if t_stop is None else int(round(t_stop * fs))
    return s[..., a:b].sum(-1).mean() / ((b - a) / fs)


def vector_strength(spikes, fs, freq, t_start=0.0):
    """Goldberg & Brown (1969) vector strength: 1 = perfect phase locking, 0 = none."""
    s = np.asarray(spikes).reshape(-1, spikes.shape[-1])
    idx = np.nonzero(s)[1]
    t = idx[idx >= t_start * fs] / fs
    if len(t) == 0:
        return np.nan
    phase = 2 * np.pi * freq * t
    return np.abs(np.mean(np.exp(1j * phase)))


def period_histogram(spikes, fs, freq, n_bins=32):
    s = np.asarray(spikes).reshape(-1, spikes.shape[-1])
    t = np.nonzero(s)[1] / fs
    phase = np.mod(t * freq, 1.0)
    counts, edges = np.histogram(phase, bins=n_bins, range=(0, 1))
    return 0.5 * (edges[1:] + edges[:-1]), counts


def isi_histogram(spikes, fs, max_isi=20e-3, bin_width=0.1e-3):
    s = np.asarray(spikes).reshape(-1, spikes.shape[-1])
    isis = np.concatenate([np.diff(np.flatnonzero(r)) / fs for r in s])
    counts, edges = np.histogram(isis, bins=int(max_isi / bin_width), range=(0, max_isi))
    return 0.5 * (edges[1:] + edges[:-1]), counts


def best_frequency(rates, freqs):
    """Frequency giving the maximal response, from a tuning curve."""
    return np.asarray(freqs)[np.argmax(rates)]

"""Plotting helpers (matplotlib)."""

import numpy as np


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    return ax


def plot_cochleagram(C, cfs, times, ax=None, cmap="magma", label="", **kw):
    ax = _ax(ax)
    im = ax.pcolormesh(times, np.arange(len(cfs)), C, shading="auto", cmap=cmap, **kw)
    ticks = np.linspace(0, len(cfs) - 1, 6).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{cfs[i]:.0f}" for i in ticks])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CF (Hz)")
    ax.figure.colorbar(im, ax=ax, label=label)
    return im


def plot_raster(spikes, fs, ax=None, color="k", markersize=2, offset=0):
    """spikes: boolean (n_trains, T)."""
    ax = _ax(ax)
    s = np.asarray(spikes).reshape(-1, spikes.shape[-1])
    for i, row in enumerate(s):
        t = np.flatnonzero(row) / fs
        ax.plot(t, np.full_like(t, i + offset), "|", color=color, markersize=markersize)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fibre / neuron")
    return ax


def log_freq_axis(ax, axis="x", ticks=(125, 250, 500, 1000, 2000, 4000, 8000)):
    """Log-scaled frequency axis with plain-number tick labels."""
    from matplotlib.ticker import NullFormatter

    getattr(ax, f"set_{axis}scale")("log")
    a = getattr(ax, f"{axis}axis")
    lo, hi = getattr(ax, f"get_{axis}lim")()
    ticks = [t for t in ticks if lo <= t <= hi]
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks])
    a.set_minor_formatter(NullFormatter())

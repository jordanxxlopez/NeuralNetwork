"""End-to-end auditory periphery: sound -> basilar membrane -> IHC -> auditory nerve."""

from dataclasses import dataclass

import numpy as np

from .auditory_nerve import AuditoryNerve
from .cochlea import GammatoneFilterbank, InnerHairCell, erb_space
from .stimuli import P_REF


@dataclass
class PeripheryOutput:
    cfs: np.ndarray
    bm: np.ndarray      # basilar membrane displacement (n_ch, T)
    ihc: np.ndarray     # inner hair cell receptor signal in [0, 1] (n_ch, T)
    rate: np.ndarray    # auditory nerve instantaneous rate (n_ch, T), spikes/s


class AuditoryPeriphery:
    """Convenience wrapper chaining the peripheral stages.

    Example
    -------
    >>> per = AuditoryPeriphery(fs=32000, n_channels=32)
    >>> out = per.process(sound)
    >>> spikes = per.spikes(out.rate, n_fibers=10)
    """

    def __init__(self, fs, n_channels=64, f_min=100.0, f_max=8000.0, cfs=None,
                 fiber_type="HSR", rng=None, ihc_kwargs=None, an_kwargs=None):
        self.fs = fs
        self.cfs = np.asarray(cfs) if cfs is not None else erb_space(f_min, f_max, n_channels)
        self.filterbank = GammatoneFilterbank(self.cfs, fs)
        self.an = AuditoryNerve(fs, fiber_type=fiber_type, rng=rng, **(an_kwargs or {}))
        self.ihc = InnerHairCell(fs, **{"p50": self.an.p50, **(ihc_kwargs or {})})

    def process(self, x):
        bm = self.filterbank.filter(x)
        ihc = self.ihc.transduce(bm)
        rate = self.an.rate(ihc)
        return PeripheryOutput(self.cfs, bm, ihc, rate)

    def spikes(self, rate, n_fibers=1):
        return self.an.spikes(rate, n_fibers)


def frame_average(x, fs, frame_rate):
    """Average a (n_ch, T) signal into non-overlapping frames -> (n_ch, n_frames)."""
    hop = int(round(fs / frame_rate))
    n_frames = x.shape[-1] // hop
    return x[..., : n_frames * hop].reshape(*x.shape[:-1], n_frames, hop).mean(-1)


def cochleagram(x, fs, n_channels=64, f_min=100.0, f_max=8000.0, frame_rate=100.0,
                kind="db", floor_db=0.0, periphery=None):
    """Auditory spectrogram, a convenient input representation for neural networks.

    kind:
      "db"   - rectified BM envelope in dB SPL (clipped at ``floor_db``)
      "rate" - auditory nerve mean firing rate (spikes/s)
    Pass a prebuilt ``periphery`` (AuditoryPeriphery) to avoid rebuilding filters
    when processing many sounds.
    Returns (cfs, frame_times, C) with C of shape (n_channels, n_frames).
    """
    per = periphery or AuditoryPeriphery(fs, n_channels, f_min, f_max)
    if kind == "db":
        bm = per.filterbank.filter(x)
        # mean of a half-wave rectified sinusoid is A/pi = rms*sqrt(2)/pi
        env = frame_average(np.maximum(bm, 0), fs, frame_rate) * np.pi / np.sqrt(2)
        C = np.maximum(20 * np.log10(env / P_REF + 1e-12), floor_db)
    elif kind == "rate":
        C = frame_average(per.process(x).rate, fs, frame_rate)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    times = (np.arange(C.shape[-1]) + 0.5) / frame_rate
    return per.cfs, times, C

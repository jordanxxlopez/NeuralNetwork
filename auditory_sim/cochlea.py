"""Cochlear mechanics: gammatone basilar-membrane filterbank and inner hair cells.

References
----------
Glasberg & Moore (1990) "Derivation of auditory filter shapes from notched-noise data."
Patterson et al. (1992) "Complex sounds and auditory images."
"""

import numpy as np
from scipy import signal


def erb_bandwidth(f):
    """Equivalent rectangular bandwidth (Hz) of the human auditory filter at ``f``."""
    return 24.7 * (4.37e-3 * np.asarray(f) + 1.0)


def hz_to_erb_rate(f):
    """Frequency (Hz) -> ERB-number (Cams)."""
    return 21.4 * np.log10(1 + 4.37e-3 * np.asarray(f))


def erb_rate_to_hz(e):
    return (10 ** (np.asarray(e) / 21.4) - 1) / 4.37e-3


def erb_space(f_min, f_max, n):
    """``n`` centre frequencies equally spaced on the ERB scale (mimics tonotopy)."""
    return erb_rate_to_hz(np.linspace(hz_to_erb_rate(f_min), hz_to_erb_rate(f_max), n))


class GammatoneFilterbank:
    """4th-order gammatone filterbank modelling basilar-membrane tuning.

    Each channel is implemented as four cascaded complex one-pole filters,
    whose impulse response is t^3 exp(-2*pi*b*t) exp(j*2*pi*cf*t), i.e. a
    gammatone. The real part is the BM velocity at that place. Gain is
    normalised to unity at each channel's characteristic frequency (CF).
    """

    def __init__(self, cfs, fs, order=4, bw_factor=1.019):
        self.cfs = np.asarray(cfs, dtype=float)
        self.fs = fs
        self.order = order
        if np.any(self.cfs >= fs / 2):
            raise ValueError("all centre frequencies must be below Nyquist")
        self._coefs = []
        for cf in self.cfs:
            bw = bw_factor * erb_bandwidth(cf)
            pole = np.exp(-2 * np.pi * bw / fs) * np.exp(2j * np.pi * cf / fs)
            a = np.poly([pole] * order)
            _, h = signal.freqz([1.0], a, worN=[cf], fs=fs)
            self._coefs.append((2.0 / np.abs(h[0]), a))  # x2: we keep the real part

    @property
    def n_channels(self):
        return len(self.cfs)

    def filter(self, x):
        """Filter a 1-D waveform -> array of shape (n_channels, n_samples)."""
        x = np.asarray(x, dtype=float)
        return np.stack([np.real(signal.lfilter([g], a, x)) for g, a in self._coefs])

    def frequency_response(self, freqs):
        """Magnitude response of every channel at ``freqs`` (n_channels, n_freqs)."""
        return np.stack([
            np.abs(signal.freqz([g], a, worN=np.asarray(freqs), fs=self.fs)[1]) / 2
            for g, a in self._coefs
        ])


class InnerHairCell:
    """Inner hair cell transduction: rectifying, saturating MET current + membrane low-pass.

    1. Mechano-electrical transduction (MET) channels open only for one direction
       of stereocilia deflection and saturate: a half-wave rectified Hill function
       of BM motion, with half-activation at ``p50`` Pa (sets the threshold).
    2. The IHC membrane acts as a low-pass filter. This is why auditory-nerve phase
       locking degrades above ~1-3 kHz: high-frequency fine structure is smoothed
       away, leaving only the envelope.

    Output is a normalised receptor signal in [0, 1].
    """

    def __init__(self, fs, p50=2e-4, hill_n=2.0, cutoff=3000.0, order=2, n_stages=2):
        self.fs = fs
        self.p50 = p50
        self.hill_n = hill_n
        self.sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
        self.n_stages = n_stages

    def transduce(self, bm):
        y = np.maximum(bm, 0.0) ** self.hill_n
        y = y / (y + self.p50 ** self.hill_n)
        for _ in range(self.n_stages):
            y = signal.sosfilt(self.sos, y, axis=-1)
        return np.clip(y, 0.0, 1.0)

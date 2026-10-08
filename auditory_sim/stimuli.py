"""Acoustic stimulus generation.

All waveforms are returned in Pascals so that sound level (dB SPL) is
meaningful to the peripheral model downstream.
"""

import numpy as np

P_REF = 20e-6  # reference pressure for dB SPL (20 µPa)


def db_to_pa(level_db):
    """Convert dB SPL to RMS pressure in Pascals."""
    return P_REF * 10 ** (np.asarray(level_db) / 20.0)


def pa_to_db(pressure):
    """Convert RMS pressure in Pascals to dB SPL."""
    return 20.0 * np.log10(np.asarray(pressure) / P_REF)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def time_vector(duration, fs):
    return np.arange(int(round(duration * fs))) / fs


def set_level(x, level_db):
    """Scale a waveform so that its RMS equals ``level_db`` dB SPL."""
    r = rms(x)
    if r == 0:
        return x.copy()
    return x * (db_to_pa(level_db) / r)


def ramp(x, fs, ramp_ms=5.0):
    """Apply raised-cosine onset/offset ramps (avoids spectral splatter)."""
    n = int(round(ramp_ms * 1e-3 * fs))
    if n == 0 or 2 * n > x.shape[-1]:
        return x
    win = np.ones(x.shape[-1])
    r = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
    win[:n] = r
    win[-n:] = r[::-1]
    return x * win


def _finish(x, fs, level_db, ramp_ms):
    return ramp(set_level(x, level_db), fs, ramp_ms)


def pure_tone(freq, duration, fs, level_db=60.0, phase=0.0, ramp_ms=5.0):
    t = time_vector(duration, fs)
    return _finish(np.sin(2 * np.pi * freq * t + phase), fs, level_db, ramp_ms)


def am_tone(carrier_freq, mod_freq, duration, fs, mod_depth=1.0, level_db=60.0, ramp_ms=5.0):
    """Sinusoidally amplitude-modulated tone."""
    t = time_vector(duration, fs)
    env = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    return _finish(env * np.sin(2 * np.pi * carrier_freq * t), fs, level_db, ramp_ms)


def harmonic_complex(f0, duration, fs, n_harmonics=10, lowest_harmonic=1, level_db=60.0,
                     phases="sine", max_freq=None, ramp_ms=5.0, rng=None):
    """Harmonic complex tone with fundamental ``f0``.

    phases: "sine", "cosine", "random", or an array of per-harmonic phases.
    Setting ``lowest_harmonic`` > 1 produces a "missing fundamental" stimulus.
    """
    t = time_vector(duration, fs)
    harmonics = np.arange(lowest_harmonic, lowest_harmonic + n_harmonics)
    nyq = fs / 2 if max_freq is None else min(max_freq, fs / 2)
    harmonics = harmonics[harmonics * f0 < nyq]
    if isinstance(phases, str):
        if phases == "sine":
            ph = np.zeros(len(harmonics))
        elif phases == "cosine":
            ph = np.full(len(harmonics), np.pi / 2)
        elif phases == "random":
            ph = np.random.default_rng(rng).uniform(0, 2 * np.pi, len(harmonics))
        else:
            raise ValueError(f"unknown phases {phases!r}")
    else:
        ph = np.asarray(phases)[: len(harmonics)]
    x = np.sin(2 * np.pi * np.outer(harmonics * f0, t) + ph[:, None]).sum(axis=0)
    return _finish(x, fs, level_db, ramp_ms)


def white_noise(duration, fs, level_db=60.0, ramp_ms=5.0, rng=None):
    x = np.random.default_rng(rng).standard_normal(len(time_vector(duration, fs)))
    return _finish(x, fs, level_db, ramp_ms)


def click_train(rate, duration, fs, click_duration=100e-6, level_db=60.0):
    """Train of rectangular clicks; level is the peak-equivalent RMS of one click."""
    x = np.zeros(len(time_vector(duration, fs)))
    width = max(1, int(round(click_duration * fs)))
    for start in np.arange(0, len(x), fs / rate).astype(int):
        x[start:start + width] = 1.0
    return x * db_to_pa(level_db)


def chirp(f_start, f_end, duration, fs, level_db=60.0, method="logarithmic", ramp_ms=5.0):
    from scipy.signal import chirp as _chirp

    t = time_vector(duration, fs)
    return _finish(_chirp(t, f_start, duration, f_end, method=method), fs, level_db, ramp_ms)


def pad(x, fs, pre=0.0, post=0.0):
    """Add silence before and after a waveform."""
    return np.concatenate([np.zeros(int(round(pre * fs))), x, np.zeros(int(round(post * fs)))])


def add_noise(x, snr_db, rng=None):
    """Add white noise at a given signal-to-noise ratio."""
    noise = np.random.default_rng(rng).standard_normal(x.shape)
    noise *= rms(x) / rms(noise) / 10 ** (snr_db / 20)
    return x + noise


def fractional_delay(x, delay, fs):
    """Delay a signal by ``delay`` seconds (may be fractional) via the FFT."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    X = np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * freqs * delay)
    return np.fft.irfft(X, nfft)[:n]


def apply_itd(x, itd, fs):
    """Create a binaural (left, right) pair with an interaural time difference.

    Convention: ``itd = t_right - t_left``. Positive ITD means the sound
    reaches the left ear first (source on the listener's left).
    """
    if itd >= 0:
        return x.copy(), fractional_delay(x, itd, fs)
    return fractional_delay(x, -itd, fs), x.copy()


def apply_ild(left, right, ild_db):
    """Interaural level difference: positive ILD makes the left ear louder."""
    g = 10 ** (ild_db / 40)
    return left * g, right / g

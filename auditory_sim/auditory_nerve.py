"""Auditory-nerve fibre (ANF) model: rate code + stochastic spike generation.

The model is phenomenological and deliberately simple so it is easy to read
and modify. It captures:
  * three spontaneous-rate classes (high/medium/low SR) with different thresholds
  * saturating rate-level functions (via the IHC, see cochlea.InnerHairCell)
  * short-term adaptation (onset peak in the PSTH)
  * phase locking (inherited from the IHC signal)
  * absolute + relative refractoriness
For a full biophysical model see Zilany, Bruce & Carney (2014).
"""

import numpy as np
from scipy import signal

from .stimuli import db_to_pa

# Spontaneous rate, peak instantaneous rate (before refractoriness) and the IHC
# half-activation pressure that sets each class's threshold. Real fibre classes
# differ at the IHC ribbon synapse; here that difference is lumped into the IHC
# sensitivity (``p50``) used for each class.
FIBER_TYPES = {
    "HSR": dict(spont=60.0, peak_rate=500.0, p50=db_to_pa(15.0)),
    "MSR": dict(spont=5.0, peak_rate=400.0, p50=db_to_pa(35.0)),
    "LSR": dict(spont=0.5, peak_rate=300.0, p50=db_to_pa(55.0)),
}


class AuditoryNerve:
    def __init__(self, fs, fiber_type="HSR", adapt_strength=0.6, adapt_tau=0.04,
                 abs_refractory=0.75e-3, rel_refractory_tau=0.6e-3, rng=None, **overrides):
        params = dict(FIBER_TYPES[fiber_type])
        params.update(overrides)
        self.fs = fs
        self.fiber_type = fiber_type
        self.spont = params["spont"]
        self.peak_rate = params["peak_rate"]
        self.p50 = params["p50"]  # used by AuditoryPeriphery to configure the IHC
        self.adapt_strength = adapt_strength
        self.adapt_tau = adapt_tau
        self.abs_refractory = abs_refractory
        self.rel_refractory_tau = rel_refractory_tau
        self.rng = np.random.default_rng(rng)

    def rate(self, ihc):
        """Instantaneous firing rate (spikes/s) from the normalised IHC signal in [0, 1]."""
        # first-order low-pass of the drive implements depletion-like adaptation
        alpha = np.exp(-1.0 / (self.adapt_tau * self.fs))
        adaptation = signal.lfilter([1 - alpha], [1, -alpha], self.adapt_strength * ihc, axis=-1)
        return self.spont + (self.peak_rate - self.spont) * ihc * (1 - adaptation)

    def spikes(self, rate, n_fibers=1):
        """Generate spikes with refractoriness.

        rate: (n_channels, n_samples) firing rate in spikes/s.
        Returns a boolean array (n_channels, n_fibers, n_samples).
        """
        rate = np.atleast_2d(rate)
        n_ch, n_t = rate.shape
        dt = 1.0 / self.fs
        out = np.zeros((n_ch, n_fibers, n_t), dtype=bool)
        since = np.full((n_ch, n_fibers), 1.0)  # time since last spike (s)
        p_base = np.clip(rate * dt, 0, 1)
        for k in range(n_t):
            rec = 1 - np.exp(-np.maximum(since - self.abs_refractory, 0) / self.rel_refractory_tau)
            s = self.rng.random((n_ch, n_fibers)) < p_base[:, k, None] * rec
            out[:, :, k] = s
            since = np.where(s, 0.0, since + dt)
        return out


def spike_times(spikes, fs):
    """Boolean spike raster (..., n_samples) -> (nested) list of spike-time arrays in s."""
    spikes = np.asarray(spikes)
    if spikes.ndim == 1:
        return np.flatnonzero(spikes) / fs
    return [spike_times(s, fs) for s in spikes]

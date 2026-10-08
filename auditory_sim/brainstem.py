"""Auditory brainstem circuits.

* Cochlear nucleus bushy cells: convergence of several ANFs onto one cell
  sharpens phase locking (Joris et al., 1994).
* Medial superior olive (MSO): Jeffress (1948) coincidence-detector array that
  maps interaural time difference (ITD) onto a place code.
"""

import numpy as np

from .neurons import LIFParams, LIFPopulation, delay, exp_synapse


class BushyCells:
    """Spherical bushy cells: each cell pools ``convergence`` ANFs from one channel."""

    def __init__(self, fs, convergence=10, tau_syn=0.3e-3, weight=None,
                 lif=LIFParams(tau_m=1e-3, refractory=0.7e-3), rng=None):
        self.fs = fs
        self.convergence = convergence
        self.tau_syn = tau_syn
        # calibrated so several near-coincident inputs are needed to fire, which
        # sharpens phase locking relative to the individual ANFs
        self.weight = weight if weight is not None else 180.0 / convergence
        self.lif = lif
        self.rng = np.random.default_rng(rng)

    def run(self, an_spikes):
        """an_spikes: (n_channels, n_fibers, T) -> bushy spikes (n_channels, n_cells, T)."""
        n_ch, n_fib, n_t = an_spikes.shape
        n_cells = max(1, n_fib // self.convergence)
        pooled = an_spikes[:, : n_cells * self.convergence].reshape(
            n_ch, n_cells, self.convergence, n_t).sum(2)
        current = exp_synapse(pooled, self.fs, self.tau_syn, self.weight)
        pop = LIFPopulation(n_ch * n_cells, self.fs, self.lif, rng=self.rng)
        spikes, _ = pop.run(current.reshape(n_ch * n_cells, n_t))
        return spikes.reshape(n_ch, n_cells, n_t)


class JeffressMSO:
    """Array of MSO coincidence detectors, each tuned to a different best ITD.

    Each neuron receives the left input delayed by max(best_itd, 0) and the right
    input delayed by max(-best_itd, 0) (ITD = t_right - t_left), so inputs arrive
    simultaneously when the stimulus ITD equals the neuron's best ITD.
    """

    def __init__(self, fs, best_itds=None, tau_syn=0.2e-3, weight=None,
                 lif=LIFParams(tau_m=0.5e-3, refractory=0.5e-3), n_inputs=20):
        self.fs = fs
        self.best_itds = np.linspace(-700e-6, 700e-6, 15) if best_itds is None else np.asarray(best_itds)
        self.tau_syn = tau_syn
        # calibrated so one ear alone rarely fires the cell (n_inputs fibres per ear)
        self.weight = weight if weight is not None else 115.0 / n_inputs
        self.lif = lif

    def input_currents(self, left_spikes, right_spikes):
        """left/right_spikes: (n_fibers, T) -> per-neuron input current (n_neurons, T)."""
        L = np.asarray(left_spikes, dtype=float).reshape(-1, left_spikes.shape[-1]).sum(0)
        R = np.asarray(right_spikes, dtype=float).reshape(-1, right_spikes.shape[-1]).sum(0)
        IL = exp_synapse(L, self.fs, self.tau_syn, self.weight)
        IR = exp_synapse(R, self.fs, self.tau_syn, self.weight)
        return np.stack([delay(IL, max(b, 0), self.fs) + delay(IR, max(-b, 0), self.fs)
                         for b in self.best_itds])

    def run(self, left_spikes, right_spikes):
        current = self.input_currents(left_spikes, right_spikes)
        pop = LIFPopulation(len(self.best_itds), self.fs, self.lif)
        spikes, v = pop.run(current)
        return spikes, v

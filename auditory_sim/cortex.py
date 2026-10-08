"""Rate-based model of tonotopic primary auditory cortex (A1).

A Wilson-Cowan network with one excitatory (E) and one inhibitory (I) population
per tonotopic channel. Excitation spreads narrowly along the frequency axis and
inhibition broadly, producing a "Mexican hat" interaction that sharpens frequency
tuning and causes lateral suppression between nearby tones.
"""

import numpy as np


def gaussian_kernel(n, sigma):
    """(n, n) row-normalised Gaussian connectivity over channel index."""
    idx = np.arange(n)
    K = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / sigma) ** 2)
    return K / K.sum(1, keepdims=True)


def relu_sat(x, r_max=1.0):
    return r_max * np.tanh(np.maximum(x, 0) / r_max)


class TonotopicCortex:
    def __init__(self, n_channels, dt=1e-3, tau_e=10e-3, tau_i=5e-3,
                 w_ee=1.2, w_ie=2.0, w_ei=1.5, w_ii=0.5,
                 sigma_e=1.5, sigma_i=6.0, input_gain=1.0, adapt_tau=0.2, adapt_strength=0.0):
        self.n = n_channels
        self.dt = dt
        self.tau_e, self.tau_i = tau_e, tau_i
        self.K_e = gaussian_kernel(n_channels, sigma_e)
        self.K_i = gaussian_kernel(n_channels, sigma_i)
        self.w_ee, self.w_ie, self.w_ei, self.w_ii = w_ee, w_ie, w_ei, w_ii
        self.input_gain = input_gain
        self.adapt_tau, self.adapt_strength = adapt_tau, adapt_strength

    def run(self, inputs):
        """inputs: (n_channels, T) thalamic drive sampled at 1/dt. Returns E, I, (n, T)."""
        n_t = inputs.shape[-1]
        E = np.zeros((self.n, n_t))
        I = np.zeros((self.n, n_t))
        e = np.zeros(self.n)
        i = np.zeros(self.n)
        a = np.zeros(self.n)  # spike-frequency adaptation of E cells
        for k in range(n_t):
            drive_e = (self.w_ee * self.K_e @ e - self.w_ie * self.K_i @ i
                       + self.input_gain * inputs[:, k] - a)
            drive_i = self.w_ei * self.K_e @ e - self.w_ii * i
            e = e + self.dt / self.tau_e * (-e + relu_sat(drive_e))
            i = i + self.dt / self.tau_i * (-i + relu_sat(drive_i))
            a = a + self.dt / self.adapt_tau * (-a + self.adapt_strength * e)
            E[:, k], I[:, k] = e, i
        return E, I

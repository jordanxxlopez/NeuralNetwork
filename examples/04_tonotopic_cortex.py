"""Tonotopic A1 as a Wilson-Cowan E/I network driven by the auditory nerve.

Shows lateral inhibition: the cortical representation of two nearby tones is
sharper than the peripheral one, and each tone suppresses its neighbours.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from auditory_sim import AuditoryPeriphery, TonotopicCortex, stimuli
from auditory_sim.periphery import frame_average
from auditory_sim.plotting import log_freq_axis

fs = 32000
n_ch = 64
per = AuditoryPeriphery(fs, n_channels=n_ch, f_min=200, f_max=8000, rng=0)

tone_a = stimuli.pure_tone(1000, 0.3, fs, 60)
tone_b = stimuli.pad(stimuli.pure_tone(1600, 0.15, fs, 60), fs, 0.1, 0.05)
sound = stimuli.pad(tone_a + tone_b, fs, 0.05, 0.1)

rate = per.process(sound).rate
drive = frame_average(rate - per.an.spont, fs, 1000) / 150.0  # thalamic input, 1 ms steps
cortex = TonotopicCortex(n_ch, dt=1e-3, adapt_strength=0.3)
E, I = cortex.run(drive)

t = np.arange(E.shape[1]) * 1e-3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
for ax, data, title in [(axes[0], drive, "Auditory nerve drive"), (axes[1], E, "A1 excitatory rate")]:
    im = ax.pcolormesh(t, per.cfs, data, shading="auto", cmap="magma")
    ax.set(title=title, xlabel="Time (s)", ylabel="CF (Hz)")
    log_freq_axis(ax, "y")
    fig.colorbar(im, ax=ax)
win = (t > 0.17) & (t < 0.28)
axes[2].plot(per.cfs, drive[:, win].mean(1) / drive[:, win].mean(1).max(), label="auditory nerve")
axes[2].plot(per.cfs, E[:, win].mean(1) / E[:, win].mean(1).max(), label="cortex (E)")
log_freq_axis(axes[2])
axes[2].set(title="Profile while both tones are on", xlabel="CF (Hz)", ylabel="Normalised response")
axes[2].legend()
save(fig, "04_tonotopic_cortex.png")

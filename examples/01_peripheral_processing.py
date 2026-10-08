"""Sound -> basilar membrane -> inner hair cells -> auditory nerve.

Visualises each stage of the periphery for a vowel-like harmonic complex.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from auditory_sim import AuditoryPeriphery, stimuli
from auditory_sim.plotting import plot_cochleagram, plot_raster
from auditory_sim.periphery import frame_average

fs = 32000
sound = stimuli.pad(stimuli.harmonic_complex(150, 0.2, fs, n_harmonics=30, level_db=65), fs, 0.02, 0.03)
per = AuditoryPeriphery(fs, n_channels=64, f_min=100, f_max=6000, rng=0)
out = per.process(sound)
spikes = per.spikes(out.rate, n_fibers=1)[:, 0]

t = np.arange(len(sound)) / fs
fig, axes = plt.subplots(4, 1, figsize=(9, 12), constrained_layout=True)
axes[0].plot(t, sound, lw=0.5)
axes[0].set(title="Stimulus: 150 Hz harmonic complex, 65 dB SPL", xlabel="Time (s)", ylabel="Pressure (Pa)")

frame_t = (np.arange(out.bm.shape[1] // int(fs / 1000)) + 0.5) / 1000
plot_cochleagram(frame_average(np.abs(out.bm), fs, 1000), per.cfs, frame_t, ax=axes[1], label="|BM| (Pa)")
axes[1].set_title("Basilar membrane (gammatone filterbank) - resolved harmonics at low CFs")
plot_cochleagram(frame_average(out.rate, fs, 1000), per.cfs, frame_t, ax=axes[2], label="spikes/s")
axes[2].set_title("Auditory nerve instantaneous rate (HSR fibres)")
plot_raster(spikes, fs, ax=axes[3])
axes[3].set_title("Auditory nerve spikes, one fibre per CF (low -> high)")
save(fig, "01_peripheral_processing.png")

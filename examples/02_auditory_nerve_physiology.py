"""Classic auditory-nerve physiology: rate-level functions, PSTH, phase locking."""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from auditory_sim import AuditoryPeriphery, analysis, stimuli
from auditory_sim.plotting import log_freq_axis

fs = 40000
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# 1. Rate-level functions for the three spontaneous-rate classes
levels = np.arange(-10, 91, 10)
for ft in ["HSR", "MSR", "LSR"]:
    per = AuditoryPeriphery(fs, cfs=[1000], fiber_type=ft, rng=1)
    rates = []
    for L in levels:
        r = per.process(stimuli.pure_tone(1000, 0.15, fs, L)).rate
        rates.append(analysis.firing_rate(per.spikes(r, 20), fs, 0.05, 0.15))
    axes[0].plot(levels, rates, "o-", label=ft)
axes[0].set(title="Rate-level functions (CF = 1 kHz)", xlabel="Level (dB SPL)", ylabel="Rate (spikes/s)")
axes[0].legend()

# 2. PSTH to a tone burst: onset peak then adaptation
per = AuditoryPeriphery(fs, cfs=[2000], rng=2)
tone = stimuli.pad(stimuli.pure_tone(2000, 0.1, fs, 60, ramp_ms=1), fs, 0.02, 0.05)
spikes = per.spikes(per.process(tone).rate, 200)
t, rate = analysis.psth(spikes, fs, bin_width=1e-3)
axes[1].bar(t * 1e3, rate, width=1.0)
axes[1].set(title="PSTH: 2 kHz tone burst (200 fibres)", xlabel="Time (ms)", ylabel="Rate (spikes/s)")

# 3. Phase locking vs frequency
freqs = np.geomspace(125, 8000, 12)
vs = []
for f in freqs:
    per = AuditoryPeriphery(fs, cfs=[f], rng=3)
    sp = per.spikes(per.process(stimuli.pure_tone(f, 0.15, fs, 60)).rate, 20)
    vs.append(analysis.vector_strength(sp[..., int(0.03 * fs):], fs, f))
axes[2].plot(freqs, vs, "o-")
log_freq_axis(axes[2])
axes[2].set(title="Phase locking (vector strength)", xlabel="Frequency (Hz)", ylabel="Vector strength", ylim=(0, 1))
save(fig, "02_auditory_nerve_physiology.png")

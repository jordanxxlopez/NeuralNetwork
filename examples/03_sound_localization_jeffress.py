"""Interaural time difference (ITD) coding in the medial superior olive.

A Jeffress delay-line array of coincidence detectors converts ITD into a place
code: the neuron whose internal delay cancels the external ITD fires the most.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from auditory_sim import AuditoryPeriphery, JeffressMSO, stimuli

fs = 40000
cf = 500.0
n_fibers = 20
per = AuditoryPeriphery(fs, cfs=[cf], rng=0)
mso = JeffressMSO(fs, best_itds=np.linspace(-600e-6, 600e-6, 9), n_inputs=n_fibers)
tone = stimuli.pure_tone(cf, 0.5, fs, 70)

stim_itds = np.linspace(-600e-6, 600e-6, 13)
counts = np.zeros((len(mso.best_itds), len(stim_itds)))
for j, itd in enumerate(stim_itds):
    left, right = stimuli.apply_itd(tone, itd, fs)
    sl = per.spikes(per.process(left).rate, n_fibers)[0]
    sr = per.spikes(per.process(right).rate, n_fibers)[0]
    spikes, _ = mso.run(sl, sr)
    counts[:, j] = spikes.sum(1) / 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for i in range(0, len(mso.best_itds), 2):
    axes[0].plot(stim_itds * 1e6, counts[i], "o-", label=f"best ITD {mso.best_itds[i] * 1e6:+.0f} µs")
axes[0].set(title=f"MSO ITD tuning curves (CF {cf:.0f} Hz)", xlabel="Stimulus ITD (µs)", ylabel="Rate (spikes/s)")
axes[0].legend(fontsize=8)
im = axes[1].imshow(counts, aspect="auto", origin="lower", cmap="viridis",
                    extent=[stim_itds[0] * 1e6, stim_itds[-1] * 1e6, mso.best_itds[0] * 1e6, mso.best_itds[-1] * 1e6])
axes[1].plot(stim_itds * 1e6, stim_itds * 1e6, "w--", lw=1)
axes[1].set(title="Place code: population response", xlabel="Stimulus ITD (µs)", ylabel="Neuron best ITD (µs)")
fig.colorbar(im, ax=axes[1], label="spikes/s")
save(fig, "03_sound_localization_jeffress.png")

decoded = mso.best_itds[counts.argmax(0)]
for s, d in zip(stim_itds, decoded):
    print(f"stimulus ITD {s * 1e6:+6.0f} µs -> decoded {d * 1e6:+6.0f} µs")

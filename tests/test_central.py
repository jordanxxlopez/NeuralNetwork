import numpy as np

from auditory_sim import (AuditoryPeriphery, BushyCells, JeffressMSO, LIFPopulation,
                          TonotopicCortex, analysis, stimuli)


def test_lif_fires_regularly_with_constant_input():
    fs = 10000
    spikes, v = LIFPopulation(1, fs).run(np.full((1, fs), 20.0))  # 20 mV > 15 mV threshold
    rate = spikes.sum()
    # analytic rate: 1 / (t_ref + tau ln(I / (I - theta)))
    expected = 1 / (1e-3 + 10e-3 * np.log(20 / 5))
    assert abs(rate - expected) / expected < 0.1
    assert LIFPopulation(1, fs).run(np.full((1, fs), 10.0))[0].sum() == 0


def test_bushy_cells_enhance_phase_locking():
    fs = 40000
    per = AuditoryPeriphery(fs, cfs=[500], rng=0)
    an = per.spikes(per.process(stimuli.pure_tone(500, 0.3, fs, 70)).rate, 40)
    bushy = BushyCells(fs, convergence=10).run(an)
    assert bushy.sum() > 0
    assert analysis.vector_strength(bushy, fs, 500) > analysis.vector_strength(an, fs, 500)


def test_mso_peaks_at_stimulus_itd():
    fs = 40000
    per = AuditoryPeriphery(fs, cfs=[500], rng=1)
    mso = JeffressMSO(fs, best_itds=np.array([-500e-6, 0, 500e-6]))
    tone = stimuli.pure_tone(500, 0.3, fs, 70)
    for target, itd in enumerate([-500e-6, 0, 500e-6]):
        left, right = stimuli.apply_itd(tone, itd, fs)
        spikes, _ = mso.run(per.spikes(per.process(left).rate, 20)[0],
                            per.spikes(per.process(right).rate, 20)[0])
        assert spikes.sum(1).argmax() == target


def test_cortex_lateral_inhibition_sharpens_profile():
    n = 40
    drive = np.zeros((n, 200))
    drive[15:25] = 1.0  # flat block of input
    E, _ = TonotopicCortex(n).run(drive)
    profile = E[:, -1]
    assert profile[15:25].max() > 0
    assert profile[10] < 0.05 * profile[15:25].max()  # suppressed outside

import numpy as np
import pytest

from auditory_sim import AuditoryPeriphery, GammatoneFilterbank, analysis, erb_space, stimuli
from auditory_sim.cochlea import erb_rate_to_hz, hz_to_erb_rate

FS = 32000


def test_level_calibration():
    x = stimuli.pure_tone(1000, 0.5, FS, level_db=70, ramp_ms=0)
    assert stimuli.pa_to_db(stimuli.rms(x)) == pytest.approx(70, abs=0.01)


def test_erb_roundtrip_and_spacing():
    f = np.array([100.0, 1000.0, 8000.0])
    assert np.allclose(erb_rate_to_hz(hz_to_erb_rate(f)), f)
    cfs = erb_space(100, 8000, 32)
    assert cfs[0] == pytest.approx(100) and cfs[-1] == pytest.approx(8000)
    assert np.all(np.diff(cfs) > 0)


@pytest.mark.parametrize("cf", [100, 1000, 6000])
def test_gammatone_unity_gain_at_cf(cf):
    fb = GammatoneFilterbank([cf], FS)
    y = fb.filter(stimuli.pure_tone(cf, 0.3, FS, ramp_ms=0) / np.sqrt(2) / stimuli.db_to_pa(60))
    assert np.abs(y[0, FS // 10:]).max() == pytest.approx(1.0, rel=0.02)


def test_tonotopy_peak_channel_matches_tone():
    per = AuditoryPeriphery(FS, n_channels=40, f_min=200, f_max=8000)
    out = per.process(stimuli.pure_tone(2000, 0.1, FS, 60))
    best = per.cfs[np.abs(out.bm).mean(1).argmax()]
    assert best == pytest.approx(2000, rel=0.1)


def test_spontaneous_rate_and_saturation():
    per = AuditoryPeriphery(FS, cfs=[1000], rng=0)
    silence = np.zeros(int(0.5 * FS))
    spont = analysis.firing_rate(per.spikes(per.process(silence).rate, 20), FS)
    driven = analysis.firing_rate(per.spikes(per.process(stimuli.pure_tone(1000, 0.5, FS, 60)).rate, 20), FS)
    assert 45 < spont < 75
    assert driven > 1.8 * spont


def test_refractory_period_respected():
    per = AuditoryPeriphery(FS, cfs=[1000], rng=0)
    sp = per.spikes(per.process(stimuli.pure_tone(1000, 0.3, FS, 80)).rate, 10)[0]
    for row in sp:
        isis = np.diff(np.flatnonzero(row)) / FS
        assert isis.size == 0 or isis.min() >= per.an.abs_refractory


def test_phase_locking_falls_with_frequency():
    vs = {}
    for f in (500, 6000):
        per = AuditoryPeriphery(FS, cfs=[f], rng=0)
        sp = per.spikes(per.process(stimuli.pure_tone(f, 0.2, FS, 60)).rate, 10)
        vs[f] = analysis.vector_strength(sp, FS, f, t_start=0.02)
    assert vs[500] > 0.3 > 0.15 > vs[6000]


def test_apply_itd_delays_right_ear():
    x = stimuli.click_train(50, 0.05, FS)
    left, right = stimuli.apply_itd(x, 500e-6, FS)
    lag = np.argmax(np.correlate(right, left, "full")) - (len(x) - 1)
    assert lag == round(500e-6 * FS)

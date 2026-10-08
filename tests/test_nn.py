import pytest

torch = pytest.importorskip("torch")

from auditory_sim.nn import CochleagramCNN, SpikingAuditoryNet, make_pitch_dataset  # noqa: E402


def test_dataset_and_models_forward_backward():
    X, y, meta = make_pitch_dataset(8, seed=0)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    n_classes = len(meta["f0s"])
    for model in (CochleagramCNN(n_classes), SpikingAuditoryNet(X.shape[1], n_classes, n_hidden=32)):
        out = model(X)
        assert out.shape == (8, n_classes)
        torch.nn.functional.cross_entropy(out, y).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_snn_returns_binary_spikes():
    X, _, _ = make_pitch_dataset(2, seed=0)
    _, spikes = SpikingAuditoryNet(X.shape[1], 4, n_hidden=16)(torch.from_numpy(X), return_spikes=True)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}

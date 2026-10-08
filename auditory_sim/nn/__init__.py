"""Neural-network models of auditory processing (requires PyTorch).

* CochleagramCNN      - convolutional "auditory cortex" trained on cochleagrams
* SpikingAuditoryNet  - recurrent spiking network (LIF) trained with surrogate gradients
* SyntheticPitchDataset - on-the-fly pitch (F0) classification task
"""

from .datasets import SyntheticPitchDataset, make_pitch_dataset
from .models import CochleagramCNN, LIFLayer, SpikingAuditoryNet, surrogate_spike
from .train import evaluate, fit

__all__ = ["CochleagramCNN", "LIFLayer", "SpikingAuditoryNet", "surrogate_spike",
           "SyntheticPitchDataset", "make_pitch_dataset", "fit", "evaluate"]

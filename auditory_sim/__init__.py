"""auditory_sim: a modular simulation of the mammalian auditory pathway.

Pathway:  sound -> cochlea (gammatone BM + IHC) -> auditory nerve
          -> cochlear nucleus (bushy cells) -> superior olive (MSO / ITD)
          -> auditory cortex (tonotopic E/I network)
          + neural-network models (auditory_sim.nn, requires PyTorch)
"""

from . import analysis, brainstem, cochlea, cortex, neurons, stimuli
from .auditory_nerve import FIBER_TYPES, AuditoryNerve, spike_times
from .brainstem import BushyCells, JeffressMSO
from .cochlea import GammatoneFilterbank, InnerHairCell, erb_space
from .cortex import TonotopicCortex
from .neurons import IzhikevichPopulation, LIFParams, LIFPopulation
from .periphery import AuditoryPeriphery, cochleagram

__version__ = "0.1.0"

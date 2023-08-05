__all__ = [
    "FFN",
    "VAE_FFN",
    "SHRED",
    "SHRED_I",
    "SHRED_III",
    "count_parameters",
]

import os
import sys
from torch.nn import Module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .FFN import FFN
from .VAE_FFN import VAE_FFN
from .SHRED import SHRED
from .SHRED_I import SHRED_I
from .SHRED_III import SHRED_III


def count_parameters(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

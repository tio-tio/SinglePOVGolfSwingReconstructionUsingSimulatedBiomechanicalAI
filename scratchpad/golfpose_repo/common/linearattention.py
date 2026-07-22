"""Stub. LinearMultiheadAttention is imported by common/model_cross.py at
module load time but is only instantiated by model variants the golfpose3d
adapter and residual-diagnostic scripts never construct (they use MixSTE2).
This stub exists purely so the top-level import in model_cross.py succeeds."""
import torch.nn as nn


class LinearMultiheadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "LinearMultiheadAttention is a stub (see linearattention.py docstring); "
            "not used by MixSTE2, which this project constructs."
        )

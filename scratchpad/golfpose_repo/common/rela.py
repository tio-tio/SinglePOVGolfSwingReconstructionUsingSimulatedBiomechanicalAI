"""Stub. RectifiedLinearAttention is imported by common/model_cross.py at
module load time but only instantiated by MixSTERELA, which the golfpose3d
adapter and residual-diagnostic scripts never construct (they use MixSTE2).
This stub exists purely so the top-level import in model_cross.py succeeds
without pulling in the real implementation's extra dependencies."""
import torch.nn as nn


class RectifiedLinearAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "RectifiedLinearAttention is a stub (see rela.py docstring); "
            "only used by MixSTERELA, which this project does not construct."
        )

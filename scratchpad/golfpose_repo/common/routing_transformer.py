"""Stub. KmeansAttention is imported by common/model_cross.py at module load
time (and its real implementation needs the `local_attention` package) but is
only instantiated by model variants (e.g. Cross_Linformer) that the golfpose3d
adapter and residual-diagnostic scripts never construct (they use MixSTE2).
This stub exists purely so the top-level import in model_cross.py succeeds."""
import torch.nn as nn


class KmeansAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "KmeansAttention is a stub (see routing_transformer.py docstring); "
            "not used by MixSTE2, which this project constructs."
        )

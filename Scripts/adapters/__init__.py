"""Model adapter registry. Each adapter wraps a pose model and emits
canonical COCO-17 landmarks per frame so they can be benchmarked against
each other in eval_utils.compute_all_metrics()."""

from .mediapipe_adapter import MediaPipeAdapter
from .yolo_adapter import YoloPoseAdapter
from .movenet_adapter import MoveNetAdapter

# tier 2 (transformer-based) - lazy import since they pull heavy deps
def _vitpose():
    from .vitpose_adapter import ViTPoseAdapter
    return ViTPoseAdapter

def _rtmpose():
    from .rtmpose_adapter import RTMPoseAdapter
    return RTMPoseAdapter

# tier 3 (3D mesh) - lazy
def _hmr2():
    from .hmr2_adapter import HMR2Adapter
    return HMR2Adapter

def _motionbert(variant: str = "lite", upstream: str = "mediapipe_heavy"):
    from .motionbert_adapter import MotionBERTAdapter
    from pathlib import Path
    return MotionBERTAdapter(
        upstream_2d_cache_dir=Path(__file__).parent.parent.parent / "Data" / "eval_runs",
        upstream_2d_model=upstream,
        device="cuda", variant=variant,
    )

def _golfpose(upstream: str = "mediapipe_heavy"):
    from .golfpose_adapter import GolfPose3DAdapter
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent
    ckpt = repo / "Models" / "golfpose" / "golfpose_17plus0.bin"
    return GolfPose3DAdapter(
        ckpt_path=ckpt,
        upstream_2d_cache_dir=repo / "Data" / "eval_runs",
        upstream_2d_model=upstream,
    )

ADAPTER_REGISTRY = {
    "mediapipe_heavy": lambda: MediaPipeAdapter(variant="heavy"),
    "mediapipe_full": lambda: MediaPipeAdapter(variant="full"),
    "mediapipe_lite": lambda: MediaPipeAdapter(variant="lite"),
    "movenet_thunder": lambda: MoveNetAdapter(variant="thunder"),
    "movenet_lightning": lambda: MoveNetAdapter(variant="lightning"),
    "yolov8n_pose": lambda: YoloPoseAdapter(variant="yolov8n-pose"),
    "yolov8m_pose": lambda: YoloPoseAdapter(variant="yolov8m-pose"),
    "vitpose_base": lambda: _vitpose()(variant="base"),
    "rtmpose_m": lambda: _rtmpose()(variant="m"),
    "hmr2": lambda: _hmr2()(),
    "motionbert_lite_from_mediapipe_heavy":  lambda: _motionbert("lite", "mediapipe_heavy"),
    "motionbert_lite_from_mediapipe_lite":   lambda: _motionbert("lite", "mediapipe_lite"),
    "motionbert_lite_from_movenet_thunder":  lambda: _motionbert("lite", "movenet_thunder"),
    "motionbert_lite_from_movenet_lightning":lambda: _motionbert("lite", "movenet_lightning"),
    "motionbert_lite_from_yolov8n_pose":     lambda: _motionbert("lite", "yolov8n_pose"),
    "motionbert_lite_from_yolov8m_pose":     lambda: _motionbert("lite", "yolov8m_pose"),
    "motionbert_lite_from_vitpose_base":     lambda: _motionbert("lite", "vitpose_base"),
    "motionbert_full_from_mediapipe_heavy":  lambda: _motionbert("full", "mediapipe_heavy"),
    "golfpose3d_from_mediapipe_heavy":       lambda: _golfpose("mediapipe_heavy"),
}

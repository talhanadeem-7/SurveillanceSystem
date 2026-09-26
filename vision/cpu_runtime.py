"""Small, explicit CPU inference controls shared by the vision components."""
import hashlib
import json
import logging
from pathlib import Path

import cv2
import torch
from torch import nn
from ultralytics.models.yolo.pose.predict import PosePredictor

import config

logger = logging.getLogger(__name__)


def cpu_pose_model_path():
    """Use only an export made from the currently configured source weights."""
    source = Path(config.MODEL_PATH)
    if getattr(config, "YOLO_CPU_BACKEND", "pytorch") != "onnx":
        return str(source)
    if getattr(config, "VISION_DEVICE", "cpu") != "cpu":
        return str(source)
    exported = Path(config.YOLO_CPU_MODEL_DIR) / (source.stem + ".onnx")
    try:
        import onnx  # noqa: F401 -- also required by Ultralytics' ONNX backend
        import onnxruntime  # noqa: F401 -- validate the optional runtime before selecting it
        metadata = json.loads(exported.with_suffix(".json").read_text())
        if (metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
                and metadata["onnx_sha256"] == hashlib.sha256(exported.read_bytes()).hexdigest()
                and metadata["precision"] == "float32" and metadata["dynamic"]):
            return str(exported)
    except (OSError, ImportError, ValueError, KeyError, TypeError):
        pass
    logger.warning("No matching CPU ONNX export; using %s. Run scripts/export_cpu_pose.py to export it.", source)
    return str(source)


class CPUPosePredictor(PosePredictor):
    """Keep Ultralytics preprocessing/postprocessing; bound ORT's CPU pool."""

    def setup_model(self, model=None, verbose=True):
        super().setup_model(model=model, verbose=verbose)
        backend = self.model.backend
        if getattr(backend, "format", None) == "onnx" and self.device.type == "cpu":
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(getattr(config, "CPU_ONNX_THREADS", 4)))
            options.inter_op_num_threads = 1
            # Idle spinning would compete with ReID/depth/face inference.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            backend.session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )


def configure_cpu_runtime():
    """Avoid competing oversized thread pools; never change the compute device."""
    threads = max(1, int(getattr(config, "CPU_TORCH_THREADS", 4)))
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    cv_threads = max(1, int(getattr(config, "CPU_OPENCV_THREADS", 1)))
    if cv2.getNumThreads() != cv_threads:
        cv2.setNumThreads(cv_threads)


def fuse_osnet_for_inference(model):
    """Fold only OSNet's known conv->BatchNorm pairs, in FP32 eval mode.

    No quantization, changed inputs, or skipped ReID evaluations. Unknown
    backbones/backends are intentionally untouched.
    """
    if not isinstance(model, nn.Module) or model.training:
        return 0
    fused = 0
    for layer in model.modules():
        if not type(layer).__module__.startswith("boxmot.reid.backbones.osnet"):
            continue
        conv_name = "conv2" if type(layer).__name__ == "LightConv3x3" else "conv"
        conv, bn = getattr(layer, conv_name, None), getattr(layer, "bn", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            if conv.training or bn.training or not bn.track_running_stats:
                continue
            setattr(layer, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            layer.bn = nn.Identity()
            fused += 1
    return fused

"""Export the configured pose weights for CPU inference. Run from the repo root.

The checkpoint is copied to a temporary directory because Ultralytics exports
beside its input. Only the deployment ONNX and provenance manifest are retained.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from ultralytics import YOLO


def main():
    source = Path(config.MODEL_PATH).resolve()
    destination = Path(config.YOLO_CPU_MODEL_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="surveillance-export-") as tmp:
        copied = Path(tmp) / source.name
        shutil.copy2(source, copied)
        exported = YOLO(str(copied), task="pose").export(
            format="onnx", dynamic=True, simplify=False, opset=17,
            device="cpu", half=False, imgsz=640, batch=1,
        )
        target = destination / (source.stem + ".onnx")
        shutil.copy2(exported, target)
    target.with_suffix(".json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "precision": "float32", "dynamic": True, "opset": 17,
    }, indent=2), encoding="utf-8")
    print(f"CPU pose model ready: {target}")


if __name__ == "__main__":
    main()

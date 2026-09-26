"""Accuracy and memory regression checks for the CPU optimizations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import config
from vision.cpu_runtime import cpu_pose_model_path, fuse_osnet_for_inference
from vision.pose_analyzer import PoseAnalyzer
from utils.heatmap import record_position, generate_heatmap, generate_trajectory


class CPUOptimizationTest(unittest.TestCase):
    def test_osnet_fusion_preserves_embeddings(self):
        from boxmot.reid.backbones.osnet import osnet_x0_25
        torch.manual_seed(7)
        model = osnet_x0_25(num_classes=10, pretrained=False).eval()
        fused = copy.deepcopy(model)
        self.assertGreater(fuse_osnet_for_inference(fused), 0)
        inputs = torch.rand(2, 3, 256, 128)
        with torch.inference_mode():
            original, optimized = model(inputs), fused(inputs)
        torch.testing.assert_close(original, optimized, rtol=1e-4, atol=1e-5)
        cosine = torch.nn.functional.cosine_similarity(original, optimized)
        self.assertTrue(torch.all(cosine > .99999))
        self.assertEqual(fuse_osnet_for_inference(fused), 0)

    def test_missing_export_falls_back_to_configured_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "YOLO_CPU_MODEL_DIR", tmp):
            self.assertEqual(cpu_pose_model_path(), config.MODEL_PATH)

    def test_mismatched_export_falls_back_to_configured_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "YOLO_CPU_MODEL_DIR", tmp):
            (Path(tmp) / "yolov8n-pose.onnx").write_bytes(b"invalid")
            (Path(tmp) / "yolov8n-pose.json").write_text('{"source_sha256":"wrong"}')
            self.assertEqual(cpu_pose_model_path(), config.MODEL_PATH)

    def test_depth_statistics_cache_invalidates_for_map_and_geometry(self):
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer._depth_stats_map = None
        analyzer._depth_stats_cache = {}
        depth = np.random.default_rng(8).random((30, 40)).astype(np.float32)
        zone = dict(type="restricted", coords=(3, 4, 18, 22))
        def expected():
            x1, y1, x2, y2 = zone['coords']
            roi = depth[y1:y2, x1:x2]
            return np.median(roi), (np.percentile(roi, 85) - np.percentile(roi, 15)) / 2
        with patch("vision.pose_analyzer.np.percentile", wraps=np.percentile) as percentile:
            first = analyzer._zone_depth_statistics(depth, zone)
            for _ in range(10):
                self.assertEqual(analyzer._zone_depth_statistics(depth, zone), first)
            self.assertEqual(percentile.call_count, 1)
        np.testing.assert_allclose(first, expected(), rtol=0, atol=0)
        zone['coords'] = (0, 0, 10, 10)
        np.testing.assert_allclose(analyzer._zone_depth_statistics(depth, zone), expected())
        depth = depth + 2
        np.testing.assert_allclose(analyzer._zone_depth_statistics(depth, zone), expected())

    def test_cached_vectorized_theft_matches_original_grid_decisions(self):
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer._theft_reference_cache = {}
        rng = np.random.default_rng(19)
        patch_image = rng.integers(0, 256, (80, 80), dtype=np.uint8)
        zone = dict(coords=(0, 0, 80, 80), reference_patch=patch_image, missing_counter=0)
        original_counter = 0
        for index in range(45):
            # Include a geometry change to exercise cache invalidation.
            width = 80 if index < 20 else 72
            zone['coords'] = (0, 0, width, 80)
            frame = (rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
                     if index % 7 == 0 else np.zeros((80, 80, 3), dtype=np.uint8))
            gray = cv2.GaussianBlur(cv2.cvtColor(frame[:, :width], cv2.COLOR_BGR2GRAY), (3, 3), 0)
            current = cv2.Canny(gray, 70, 200)
            reference = cv2.Canny(cv2.resize(patch_image, (width, 80)), 70, 200)
            gh, gw = 80 // config.GRID_SIZE, width // config.GRID_SIZE
            missing = 0
            for row in range(config.GRID_SIZE):
                for col in range(config.GRID_SIZE):
                    ys, xs = slice(row * gh, (row + 1) * gh), slice(col * gw, (col + 1) * gw)
                    ref, cur = np.count_nonzero(reference[ys, xs]), np.count_nonzero(current[ys, xs])
                    missing += int(ref > 10 and (ref - cur) / ref * 100 > config.THEFT_THRESHOLD)
            original_counter = original_counter + 1 if missing >= 2 else 0
            result = analyzer.detect_theft(frame, zone)
            self.assertEqual(zone['missing_counter'], original_counter)
            self.assertEqual(result, original_counter > config.THEFT_FRAME_PERSISTENCE)

    def test_heatmap_retention_keeps_latest_points_and_still_renders(self):
        store = {}
        with patch.object(config, "HEATMAP_MAX_TRACKS", 2), patch.object(config, "HEATMAP_MAX_POINTS_PER_TRACK", 3):
            for frame in range(5):
                record_position(store, 1, np.array([0, 0, 10, 10]), frame)
            record_position(store, 2, np.array([0, 0, 10, 10]), 5)
            record_position(store, 3, np.array([0, 0, 10, 10]), 6)
        self.assertNotIn(1, store)
        self.assertEqual(set(store), {2, 3})
        reference = np.zeros((40, 40, 3), np.uint8)
        self.assertEqual(generate_heatmap(store[2], reference).shape, reference.shape)
        self.assertEqual(generate_trajectory(store[2], reference).shape, reference.shape)

    def test_grab_retrieve_preserves_selected_frames(self):
        path = Path(config.VIDEO_DIR) / 'sample12.mp4'
        if not path.exists():
            self.skipTest("Local sample12 is not present")
        old, new = cv2.VideoCapture(str(path)), cv2.VideoCapture(str(path))
        try:
            for index in range(30):
                ok, frame = old.read()
                self.assertEqual(ok, new.grab())
                if not ok:
                    break
                if (index + 1) % 2 == 0:
                    retrieved, selected = new.retrieve()
                    self.assertTrue(retrieved)
                    np.testing.assert_array_equal(frame, selected)
        finally:
            old.release()
            new.release()


if __name__ == "__main__":
    unittest.main()

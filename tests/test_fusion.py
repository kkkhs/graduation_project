from __future__ import annotations

import unittest

from src.application.fusion import fuse_predictions, iou_xywh


class FusionTests(unittest.TestCase):
    def test_iou_xywh_basic(self) -> None:
        a = [0, 0, 10, 10]
        b = [5, 5, 10, 10]
        iou = iou_xywh(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_fuse_predictions_weighted_cluster(self) -> None:
        records = [
            {"image_id": "a.jpg", "model_name": "drenet", "bbox": [100, 100, 20, 20], "score": 0.9, "category_id": 0, "inference_time": 12.0},
            {"image_id": "a.jpg", "model_name": "mmdet", "bbox": [102, 100, 20, 20], "score": 0.8, "category_id": 0, "inference_time": 11.0},
            {"image_id": "a.jpg", "model_name": "yolo", "bbox": [101, 99, 20, 20], "score": 0.7, "category_id": 0, "inference_time": 10.0},
        ]

        fused = fuse_predictions(records, iou_threshold=0.3, min_votes=1)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["model_name"], "ensemble")
        self.assertGreater(fused[0]["score"], 0.7)

    def test_fuse_predictions_min_votes_filter(self) -> None:
        records = [
            {"image_id": "a.jpg", "model_name": "drenet", "bbox": [0, 0, 10, 10], "score": 0.9, "category_id": 0, "inference_time": 1.0},
            {"image_id": "a.jpg", "model_name": "mmdet", "bbox": [80, 80, 10, 10], "score": 0.8, "category_id": 0, "inference_time": 1.0},
        ]
        fused = fuse_predictions(records, iou_threshold=0.5, min_votes=2)
        self.assertEqual(fused, [])


if __name__ == "__main__":
    unittest.main()

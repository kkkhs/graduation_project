from __future__ import annotations

import unittest


class CompatImportTests(unittest.TestCase):
    def test_core_predictor_import(self) -> None:
        from src.core.predictor import UnifiedPredictor

        self.assertEqual(UnifiedPredictor.__name__, "UnifiedPredictor")

    def test_adapter_import(self) -> None:
        from src.adapters.yolo_adapter import YOLOAdapter

        self.assertEqual(YOLOAdapter.__name__, "YOLOAdapter")


if __name__ == "__main__":
    unittest.main()

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import app as backend_app  # noqa: E402


class BackendSmokeTests(unittest.TestCase):
    def test_health_reports_loaded_models(self):
        result = backend_app.health()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["models_loaded"])
        self.assertTrue(result["opportunity_analysis"])

    def test_artifact_schema_is_consistent(self):
        config = backend_app.get_config()
        self.assertTrue(config["facilities"])
        analysis = backend_app.get_opportunities()
        self.assertEqual(analysis["method"]["folds"], 3)
        self.assertGreater(len(analysis["facilities"]), 1)
        self.assertTrue(analysis["opportunities"])
        self.assertIn("los_mae", analysis["validation"])
        first = analysis["opportunities"][0]
        self.assertIn("robust_net_cost_difference", first)
        self.assertIn("top_10_positive_cost_share", first)
        self.assertTrue(first["fdr_significant"])
        self.assertLessEqual(first["q_value"], 0.05)
        self.assertTrue(
            "APR DRG Description" in first or "Service Line" in first)
        self.assertIn(first["signal_pattern"], {
            "broad-based", "mixed", "outlier-concentrated"})
        for feature in ("DRG Severity Group", "Admission Pathway",
                        "Population Group", "Complexity Score"):
            self.assertIn(feature, backend_app.CAT)

    def test_prediction_returns_finite_values_and_drivers(self):
        options = backend_app.get_config()["ui_options"]
        request = backend_app.PredictIn(
            age=options["Age Group"][0],
            admission=options["Type of Admission"][0],
            severity=options["APR Severity of Illness Description"][0],
            drg=options["APR DRG Description"][0],
            payer=options["Payment Typology 1"][0],
        )
        result = backend_app.predict(request)
        self.assertGreaterEqual(result["los"], 0.5)
        self.assertGreaterEqual(result["cost"], 0)
        self.assertIsInstance(result["high_cost"], bool)
        self.assertGreaterEqual(len(result["drivers"]), 10)

    def test_artifacts_are_valid_json(self):
        for name in ("config.json", "metrics.json"):
            with (ROOT / "backend" / "artifacts" / name).open() as handle:
                self.assertIsInstance(json.load(handle), dict)


if __name__ == "__main__":
    unittest.main()

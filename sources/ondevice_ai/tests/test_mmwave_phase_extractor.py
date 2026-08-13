import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mmwave_phase_extractor import (  # noqa: E402
    MmwavePhaseExtractor,
    PROFILE_ID,
    PhaseExtractionError,
    SearchRegion,
    array_sha256,
    deterministic_argmax,
)
from validate_mmwave_phase_pilot import derive_gate, validate_manifests  # noqa: E402


def fixture(frames=200, channels=8, bins=8):
    t = np.arange(frames) / 10.0
    values = np.full((frames, channels, bins), 0.1 + 0.0j, dtype=np.complex128)
    for channel in range(channels):
        values[:, channel, 2] = (2 + channel / 10) * np.exp(
            1j * (0.8 * np.sin(2 * np.pi * 0.25 * t) + channel / 5)
        )
    return values, np.arange(bins, dtype=np.float64) * 0.31724069629629614


class TestPhaseExtractor(unittest.TestCase):
    def setUp(self):
        self.values, self.rbins = fixture()
        self.region = SearchRegion("PILOT_SEARCH_REGION_001", 0.3, 1.91)
        self.extractor = MmwavePhaseExtractor(self.region)

    def test_01_deterministic_bin_selection(self):
        a = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        b = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        self.assertEqual(a["selection"], b["selection"])
        self.assertEqual(a["selection"]["selected_range_bin_index"], 2)

    def test_02_tie_breaking(self):
        self.assertEqual(deterministic_argmax([(1.0, 3, 2), (1.0, 2, 7)]), (1.0, 2, 7))
        self.assertEqual(deterministic_argmax([(1.0, 2, 7), (1.0, 2, 1)]), (1.0, 2, 1))

    def test_03_search_region_enforcement(self):
        result = self.extractor.analyze_candidates(self.values, self.rbins)
        self.assertTrue(all(1 <= index <= 6 for index in result["eligible_bin_indices"]))

    def test_04_out_of_range_region_rejected(self):
        with self.assertRaises(PhaseExtractionError):
            SearchRegion("EMPTY", 30, 40).eligible_indices(self.rbins)

    def test_05_single_channel_extraction(self):
        result = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        self.assertEqual(result["complex_signal"].shape, (200,))
        self.assertEqual(len(result["selection"]["selected_virtual_channels"]), 1)

    def test_06_multichannel_candidates_are_evaluated(self):
        strategies = self.extractor.analyze_candidates(self.values, self.rbins)["strategy_results"]
        names = {row["virtual_channel_strategy"] for row in strategies}
        self.assertIn("V2_QUALITY_WEIGHTED_PHASE", names)
        self.assertIn("V3_MEDIAN_CONSENSUS_PHASE", names)

    def test_07_wrapped_phase_matches_numpy_angle(self):
        result = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        self.assertTrue(np.array_equal(result["wrapped_phase"], np.angle(result["complex_signal"])))

    def test_08_unwrap_matches_numpy(self):
        result = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        self.assertTrue(np.array_equal(result["unwrapped_phase"], np.unwrap(result["wrapped_phase"])))

    def test_09_near_zero_magnitude_flag(self):
        values = self.values.copy()
        values[10, :, 2] = 0
        result = self.extractor.extract(rffts=values, rbins=self.rbins)
        self.assertGreater(result["statistics"]["near_zero_magnitude_count"], 0)
        self.assertIn("NEAR_ZERO_MAGNITUDE_SAMPLES_PRESERVED", result["warnings"])

    def test_10_nonfinite_is_flagged(self):
        values = self.values.copy()
        values[10, :, 2] = complex(np.nan, 0)
        result = self.extractor.extract(rffts=values, rbins=self.rbins)
        self.assertIn("NONFINITE_CANDIDATES_EXCLUDED_FROM_ENERGY_SELECTION", result["warnings"])

    def test_11_candidate_scoring_is_deterministic(self):
        a = self.extractor.analyze_candidates(self.values, self.rbins)
        b = self.extractor.analyze_candidates(self.values, self.rbins)
        self.assertEqual(a["selected_bin_index"], b["selected_bin_index"])
        self.assertEqual(a["selected_channel"], b["selected_channel"])
        self.assertEqual(a["candidate_metrics"], b["candidate_metrics"])

    def test_12_selection_api_has_no_labels(self):
        base = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        annotated = self.extractor.extract(rffts=self.values, rbins=self.rbins, annotation={"label": "APNEA"})
        self.assertEqual(base["selection"], annotated["selection"])
        self.assertTrue(base["selection"]["label_independent"])

    def test_13_stored_rbins_are_used(self):
        result = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        index = result["selection"]["selected_range_bin_index"]
        self.assertEqual(result["selection"]["selected_range_m"], self.rbins[index])

    def test_14_config_rbin_does_not_override_stored_rbins(self):
        result = self.extractor.extract(
            rffts=self.values, rbins=self.rbins, config={"R_BIN": 99.0}
        )
        self.assertNotEqual(result["selection"]["selected_range_m"], 99.0 * 2)

    def test_15_canonical_phase_is_not_diagnostic_filtered(self):
        result = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        expected = np.unwrap(np.angle(result["complex_signal"]))
        self.assertTrue(np.array_equal(result["unwrapped_phase"], expected))

    def test_16_annotation_cannot_modify_selection(self):
        first = self.extractor.extract(rffts=self.values, rbins=self.rbins, annotation=[1, 2])
        second = self.extractor.extract(rffts=self.values, rbins=self.rbins, annotation=[100])
        self.assertEqual(first["selection"], second["selection"])

    def test_17_validator_failure_forces_nonready_gate(self):
        self.assertEqual(derive_gate(validation_success=False, failure_count=0, warning_count=0), ("FAIL", "NOT_READY"))

    def test_18_deterministic_artifact_regeneration(self):
        a = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        b = self.extractor.extract(rffts=self.values, rbins=self.rbins)
        self.assertEqual(array_sha256(a["unwrapped_phase"]), array_sha256(b["unwrapped_phase"]))

    def test_19_invalid_structural_shape_rejected(self):
        with self.assertRaises(PhaseExtractionError):
            self.extractor.extract(rffts=self.values[:, 0, :], rbins=self.rbins)

    def test_20_validator_rejects_outside_search_region(self):
        pilot = {"recordings": [{"recording_id": "r1"}]}
        selected = [{
            "recording_id": "r1", "selected_range_bin_index": 7,
            "selected_virtual_channels": [0], "selection_used_labels": False,
            "selected_extraction_profile": PROFILE_ID, "canonical_phase_length": 10,
            "frame_count": 10, "timestamp_count": 10, "nonfinite_phase_count": 0,
            "quality_status": "SUCCESS", "warnings": [], "errors": [],
        }]
        result = validate_manifests(
            pilot_selection=pilot, candidate_results=[{"recording_id": "r1"}],
            selected_results=selected,
            search_region={"eligible_bin_indices": [1, 2], "stored_rbins_count": 8},
            profiles_doc={"profiles": [{"profile_id": PROFILE_ID}]},
            exceptions_doc={"exceptions": []}, valid_decoded_recording_ids={"r1"},
        )
        self.assertFalse(result["validation_success"])


if __name__ == "__main__":
    unittest.main()

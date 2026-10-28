#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    RetinalConcept-Net
    Research code: Ali Golbaf

    Task:
        Dataset-free checks for experiment arguments and reporting.
"""

import contextlib
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from cli_config import parse_arguments
from reporting import calculate_per_class_metrics, expected_calibration_error


class ExperimentArgumentsTests(unittest.TestCase):

    def test_original_defaults(self):

        args = parse_arguments(["--mode", "Train_CBM", "--data-root", "sample-data"])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.num_folds, 5)
        self.assertEqual(args.batch_size, 32)
        self.assertTrue(args.load_existing_models)

    def test_optional_flags(self):

        args = parse_arguments(["--mode", "Test_CBM", "--data-root", "sample-data",
                                "--fold", "2", "--device", "cpu",
                                "--no-load-existing-models"])
        self.assertEqual(args.fold, 2)
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.load_existing_models)

    def test_invalid_fold_rejected(self):

        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_arguments(["--mode", "Train_CBM", "--data-root", "sample-data",
                                 "--fold", "6"])

    def test_dry_run_without_pytorch(self):

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "img").mkdir()
            (folder / "FP.csv").write_text("image,grade,lesion\n", encoding="utf-8")
            with contextlib.redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as context:
                    parse_arguments(["--mode", "Train_Black-box",
                                     "--data-root", str(folder), "--dry-run"])
            self.assertEqual(context.exception.code, 0)


class ReportingTests(unittest.TestCase):

    def test_per_class_and_underestimation(self):

        confusion = [[2, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]]
        per_class, underestimations = calculate_per_class_metrics(confusion)
        self.assertEqual(underestimations, 1)
        self.assertEqual(per_class["0"]["support"], 2)
        self.assertAlmostEqual(per_class["1"]["recall"], 0.5)

    def test_expected_calibration_error(self):

        value = expected_calibration_error([0.75, 0.25], [True, False])
        self.assertAlmostEqual(value, 0.25)


if __name__ == "__main__":
    unittest.main()

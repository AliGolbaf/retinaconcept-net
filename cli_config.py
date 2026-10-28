#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    RetinalConcept-Net
    Research code: Ali Golbaf

    Task:
        Command-line arguments for the original DR grading experiments.
"""

################################################################################
################################################################################
""" Libraries """
import argparse
from pathlib import Path


################################################################################
################################################################################
""" Experiment arguments """

def parse_arguments(argv=None):

    parser = argparse.ArgumentParser(
        description="RetinalConcept-Net: black-box and CBM DR grading using MMRDR-CFP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment ---------------------------------------------------------------
    parser.add_argument("--mode", required=True,
                        choices=["Train_Black-box", "Train_CBM",
                                 "Test_Black-box", "Test_CBM"])
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to the MMRDR-CFP directory.")
    parser.add_argument("--annotations-file", default="FP.csv",
                        help="Annotation CSV filename inside the data root.")
    parser.add_argument("--images-subdir", default="img",
                        help="Image subdirectory inside the data root.")
    parser.add_argument("--output-dir", default="outputs", type=Path)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="Optional 1-based fold number; omit to process all folds.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, nargs=2,
                        metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Optimisation -------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--load-existing-models", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--blackbox-patience", type=int, default=10)

    # Two-stage CBM training ---------------------------------------------------
    parser.add_argument("--concept-epochs", type=int, default=100)
    parser.add_argument("--grading-epochs", type=int, default=100)
    parser.add_argument("--concept-patience", type=int, default=10)
    parser.add_argument("--grading-patience", type=int, default=10)

    # Reproducibility ----------------------------------------------------------
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check arguments and paths without importing PyTorch.")

    args = parser.parse_args(argv)

    if args.num_folds < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("Fold count and batch size must be positive; workers cannot be negative.")
    if args.mode.startswith("Train_") and args.num_folds < 2:
        parser.error("Training cross-validation requires at least two folds.")
    if args.fold is not None and not 1 <= args.fold <= args.num_folds:
        parser.error("--fold must be between 1 and --num-folds.")
    if any(size < 32 for size in args.image_size):
        parser.error("Each image dimension must be at least 32 pixels.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative.")
    for name in ("epochs", "blackbox_patience", "concept_epochs",
                 "grading_epochs", "concept_patience", "grading_patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.dry_run:
        annotations = args.data_root / args.annotations_file
        images = args.data_root / args.images_subdir
        if not annotations.is_file():
            parser.error(f"Annotation file not found: {annotations}")
        if not images.is_dir():
            parser.error(f"Image folder not found: {images}")
        print("Arguments and dataset paths verified. No model has been loaded.")
        for name, value in vars(args).items():
            print(f"{name}: {value}")
        raise SystemExit(0)

    return args

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Research project: Ali Golbaf

    Task:
        Independent MMRDR-CFP test-set evaluation and DR error analysis.

    Note:
        Results are research outputs, not calibrated clinical predictions.
"""

################################################################################
################################################################################
""" Libraries """
import argparse
import json
from pathlib import Path


################################################################################
################################################################################
""" Test configuration """

def parse_arguments():

    parser = argparse.ArgumentParser(description="Independent DR grading case-level evaluation.")
    parser.add_argument("--model", required=True, choices=["blackbox", "cbm"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/evaluation"), type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--image-size", nargs=2, type=int, default=(224, 224))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0 or min(args.image_size) < 32:
        parser.error("Invalid batch size, workers or image size.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    if not (args.data_root / "FP.csv").is_file():
        parser.error(f"Annotation file not found: {args.data_root / 'FP.csv'}")
    return args


################################################################################
################################################################################
""" Test model """

def main():

    args = parse_arguments()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from data_loader import SubjectsDataset
    from models import BlackBox_Model, Concept_Bottleneck_Model
    from reporting import calculate_per_class_metrics, expected_calibration_error
    from train import calculate_grading_metrics
    from utils import DR_GRADES, load_data, read_csv_exl

    device_name = ("cuda" if args.device == "cuda" or
                   (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(device_name)

    data_clinical = read_csv_exl(str(args.data_root), "FP.csv")
    data = load_data(str(args.data_root / "img"),
                     data_clinical["image"].values,
                     data_clinical["grade"].values,
                     data_clinical["lesion"].values)
    data_test = [subject for subject in data
                 if subject["subject_id"].startswith("ts")]
    if not data_test:
        raise ValueError("No ts* records found in MMRDR-CFP.")

    test_loader = DataLoader(SubjectsDataset(data=data_test,
                                            image_size=tuple(args.image_size),
                                            augment=False),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    if args.model == "cbm":
        model = Concept_Bottleneck_Model(num_grade_classes=5,
                                         num_concepts=7,
                                         pretrained=False).to(device)
    else:
        model = BlackBox_Model(num_grade_classes=5,
                               pretrained=False).to(device)

    model.load_state_dict(torch.load(args.checkpoint,
                                     map_location="cpu",
                                     weights_only=True), strict=True)
    model.eval()

    targets = []
    predictions = []
    confidences = []
    correct = []
    cases = []

    with torch.inference_mode():
        for subject_ids, images, grades, _ in test_loader:

            images = images.to(device)
            true_grades = grades.argmax(dim=1).cpu().tolist()

            if args.model == "cbm":
                outputs = model(images, return_maps=False)
            else:
                outputs = model(images)

            grade_scores = torch.softmax(outputs["grade_logits"], dim=1)
            predicted_grades = grade_scores.argmax(dim=1).cpu().tolist()
            probabilities = grade_scores.cpu().tolist()

            concept_probabilities = None
            if args.model == "cbm":
                concept_probabilities = outputs["concept_probs"].cpu().tolist()

            for index, subject_id in enumerate(subject_ids):

                actual = int(true_grades[index])
                predicted = int(predicted_grades[index])
                confidence = float(max(probabilities[index]))

                targets.append(actual)
                predictions.append(predicted)
                confidences.append(confidence)
                correct.append(actual == predicted)

                case = {"image_id": str(subject_id),
                        "true_grade": actual,
                        "predicted_grade": predicted,
                        "predicted_label": DR_GRADES[predicted],
                        "grade_scores": probabilities[index],
                        "top_class_score": confidence,
                        "underestimation": predicted < actual}
                if concept_probabilities is not None:
                    case["concept_scores"] = concept_probabilities[index]
                cases.append(case)

    metrics = calculate_grading_metrics(targets, predictions)
    per_class, underestimation_count = calculate_per_class_metrics(
        metrics["confusion_matrix"])
    metrics["per_class"] = per_class
    metrics["underestimation_count"] = underestimation_count
    metrics["ece_10_bins"] = expected_calibration_error(confidences, correct)
    metrics["mean_top_class_score"] = float(np.mean(confidences))
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.model}_case_analysis.json"
    with path.open("w", encoding="utf-8") as output_file:
        json.dump({"model": args.model,
                   "checkpoint": args.checkpoint.name,
                   "number_of_test_images": len(cases),
                   "metrics": metrics,
                   "cases": cases,
                   "warning": "Softmax scores are not calibrated clinical probabilities."},
                  output_file, indent=2)

    print(f"Independent test images: {len(cases)}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | "
          f"Macro-F1: {metrics['macro_f1']:.4f} | "
          f"ECE: {metrics['ece_10_bins']:.4f}")
    print(f"Results saved: {path}")


if __name__ == "__main__":
    main()

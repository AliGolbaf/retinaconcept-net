#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    RetinalConcept-Net
    Research code: Ali Golbaf

    Task:
        Save model evaluation results with per-class DR grading metrics.
"""

################################################################################
################################################################################
""" Libraries """
import json
import os
import statistics


################################################################################
################################################################################
""" Results """

def _to_standard_type(value):
    """Convert NumPy outputs into values accepted by JSON."""

    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def calculate_per_class_metrics(confusion_matrix):
    """Use rows=true grades and columns=predicted grades."""

    matrix = _to_standard_type(confusion_matrix)
    number_of_classes = len(matrix)
    results = {}

    for class_index in range(number_of_classes):

        true_positive = matrix[class_index][class_index]
        actual_count = sum(matrix[class_index])
        predicted_count = sum(matrix[row][class_index]
                              for row in range(number_of_classes))

        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)

        results[str(class_index)] = {"precision": precision,
                                     "recall": recall,
                                     "f1": f1,
                                     "support": actual_count}

    underestimation_count = sum(matrix[actual][predicted]
                                for actual in range(number_of_classes)
                                for predicted in range(actual))

    return results, underestimation_count


def expected_calibration_error(confidences, correct, number_of_bins=10):
    """Top-class ECE; it measures miscalibration but does not correct it."""

    if len(confidences) != len(correct) or not confidences:
        raise ValueError("Confidence and correctness arrays must have equal nonzero length.")

    result = 0.0
    for bin_index in range(number_of_bins):

        lower = bin_index / number_of_bins
        upper = (bin_index + 1) / number_of_bins
        indices = [index for index, confidence in enumerate(confidences)
                   if lower <= confidence < upper or
                   (bin_index == number_of_bins - 1 and confidence == upper)]

        if indices:
            average_confidence = sum(confidences[index] for index in indices) / len(indices)
            accuracy = sum(bool(correct[index]) for index in indices) / len(indices)
            result += (len(indices) / len(confidences)) * abs(average_confidence - accuracy)

    return result


def save_results(path_to_outputs, mode, configuration, fold_details):
    """Save the experiment metrics without changing the original training loops."""

    if not fold_details:
        raise ValueError("Cannot save results without evaluated folds.")

    output = []
    for result in fold_details:

        metrics = {key: _to_standard_type(value)
                   for key, value in result["metrics"].items()}
        per_class, underestimations = calculate_per_class_metrics(
            metrics["confusion_matrix"])

        metrics["per_class"] = per_class
        metrics["underestimation_count"] = underestimations
        output.append({"fold": result["fold"], "metrics": metrics})

    summary = {}
    for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "quadratic_kappa"):

        values = [float(result["metrics"][metric_name]) for result in output]
        summary[metric_name] = {"mean": statistics.mean(values),
                                "std": statistics.stdev(values) if len(values) > 1 else 0.0}

    path = os.path.join(path_to_outputs, f"{mode.lower()}_metrics.json")
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump({"mode": mode,
                   "configuration": configuration,
                   "per_fold": output,
                   "summary": summary}, output_file, indent=2)

    print(f"Metrics saved: {path}")

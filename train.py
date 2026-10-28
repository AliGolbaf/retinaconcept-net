#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Written by: Ali Golbaf (ali.golbaf71@gmail.com)

    Tasks:
        * Training and validation functions for black-box DR grading.
        * Training and validation functions for CBM DR grading.

"""
################################################################################
################################################################################
""" Libraries """
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             cohen_kappa_score,
                             f1_score,
                             confusion_matrix)


################################################################################
################################################################################
""" General functions """

def calculate_grading_metrics(targets, predictions, num_grade_classes=5):
    """Calculate grading metrics, with balanced accuracy as the primary metric."""

    labels = list(range(num_grade_classes))

    return {"balanced_accuracy": balanced_accuracy_score(targets, predictions),
            "accuracy": accuracy_score(targets, predictions),
            "macro_f1": f1_score(targets, predictions,
                                 labels=labels, average="macro", zero_division=0),
            "quadratic_kappa": cohen_kappa_score(targets, predictions,
                                                  labels=labels, weights="quadratic"),
            "confusion_matrix": confusion_matrix(targets, predictions,
                                                 labels=labels)}


def calculate_concept_metrics(targets, predictions):
    """Calculate binary BAC for each concept and their mean."""

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    concept_bac = []

    for concept_index in range(targets.shape[1]):

        true_labels = targets[:, concept_index]
        predicted_labels = predictions[:, concept_index]

        if np.unique(true_labels).size == 2:
            concept_bac.append(balanced_accuracy_score(true_labels,
                                                       predicted_labels))
        else:
            concept_bac.append(float("nan"))

    concept_bac = np.asarray(concept_bac, dtype=np.float32)

    return {"concept_balanced_accuracy": float(np.nanmean(concept_bac)),
            "concept_bac_per_class": concept_bac}


################################################################################
################################################################################
""" Black-box Training """

def train_one_epoch_blackbox(model,
                             data_loader,
                             optimiser,
                             loss_fn,
                             device,
                             scaler=None,
                             use_amp=False,
                             max_grad_norm=1.0):
    """Train one epoch for five-class DR grading."""

    model.train()
    running_loss = 0.0
    number_of_images = 0
    targets = []
    predictions = []

    for _, image, grade, _ in tqdm(data_loader, desc="Training", unit="batch"):

        image = image.to(device, non_blocking=True)
        grade = grade.argmax(dim=1).long().to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        # Prediction and loss --------------------------------------------------
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(image)
            loss = loss_fn(outputs["grade_logits"], grade)

        # Backpropagation ------------------------------------------------------
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()

        running_loss += loss.item() * image.size(0)
        number_of_images += image.size(0)
        targets.extend(grade.cpu().tolist())
        predictions.extend(outputs["grade_logits"].argmax(dim=1)
                           .detach().cpu().tolist())

    metrics = calculate_grading_metrics(targets, predictions)
    metrics["loss"] = running_loss / number_of_images
    return metrics


################################################################################
################################################################################
""" Black-box Validation """

@torch.inference_mode()
def validate_one_epoch_blackbox(model,
                                data_loader,
                                loss_fn,
                                device,
                                num_grade_classes=5,
                                use_amp=False):
    """Validate one epoch for five-class DR grading."""

    model.eval()
    running_loss = 0.0
    number_of_images = 0
    targets = []
    predictions = []

    for _, image, grade, _ in tqdm(data_loader, desc="Validation", unit="batch"):

        image = image.to(device, non_blocking=True)
        grade = grade.argmax(dim=1).long().to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(image)
            loss = loss_fn(outputs["grade_logits"], grade)

        running_loss += loss.item() * image.size(0)
        number_of_images += image.size(0)
        targets.extend(grade.cpu().tolist())
        predictions.extend(outputs["grade_logits"].argmax(dim=1)
                           .cpu().tolist())

    metrics = calculate_grading_metrics(targets, predictions, num_grade_classes)
    metrics["loss"] = running_loss / number_of_images
    return metrics


################################################################################
################################################################################
""" CBM Training """

def train_one_epoch_cbm(model,
                        data_loader,
                        optimiser,
                        grade_loss_fn,
                        concept_loss_fn,
                        device,
                        grade_loss_weight=1.0,
                        concept_loss_weight=0.1,
                        scaler=None,
                        use_amp=False,
                        max_grad_norm=1.0):
    """Train the CBM using concept-only or joint supervision."""

    # Set the training mode -----------------------------------------------
    if grade_loss_weight == 0.0:
        model.train()
    else:
        model.eval()
        model.grade_head.train()
        
    running_loss = 0.0
    running_grade_loss = 0.0
    running_concept_loss = 0.0
    number_of_images = 0
    targets = []
    predictions = []
    concept_targets = []
    concept_predictions = []

    for _, image, grade, concepts in tqdm(data_loader,
                                          desc="Training CBM", unit="batch"):

        image = image.to(device, non_blocking=True)
        grade = grade.argmax(dim=1).long().to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)

        # Prediction and losses ------------------------------------------------
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(image, return_maps=False)
            grade_loss = grade_loss_fn(outputs["grade_logits"], grade)
            concept_loss = concept_loss_fn(outputs["concept_logits"], concepts)

            # Keep the grading head outside concept-only optimisation.
            if grade_loss_weight == 0.0:
                loss = concept_loss_weight * concept_loss
            else:
                loss = (grade_loss_weight * grade_loss
                        + concept_loss_weight * concept_loss)

        # Backpropagation ------------------------------------------------------
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()

        # Losses and predictions -----------------------------------------------
        batch_size = image.size(0)
        running_loss += loss.item() * batch_size
        running_grade_loss += grade_loss.item() * batch_size
        running_concept_loss += concept_loss.item() * batch_size
        number_of_images += batch_size

        targets.extend(grade.cpu().tolist())
        predictions.extend(outputs["grade_logits"].argmax(dim=1)
                           .detach().cpu().tolist())
        concept_targets.extend(concepts.cpu().tolist())
        concept_predictions.extend((outputs["concept_logits"].float()
                                    .sigmoid() >= 0.5).int().detach().cpu().tolist())

    metrics = calculate_grading_metrics(targets, predictions)
    metrics.update(calculate_concept_metrics(concept_targets, concept_predictions))
    metrics.update({"loss": running_loss / number_of_images,
                    "grade_loss": running_grade_loss / number_of_images,
                    "concept_loss": running_concept_loss / number_of_images})
    return metrics


################################################################################
################################################################################
""" CBM Validation """

@torch.inference_mode()
def validate_one_epoch_cbm(model,
                           data_loader,
                           grade_loss_fn,
                           concept_loss_fn,
                           device,
                           num_grade_classes=5,
                           grade_loss_weight=1.0,
                           concept_loss_weight=0.1,
                           use_amp=False):
    """Validate CBM grading and concept predictions."""

    model.eval()
    running_loss = 0.0
    running_grade_loss = 0.0
    running_concept_loss = 0.0
    number_of_images = 0
    targets = []
    predictions = []
    concept_targets = []
    concept_predictions = []

    for _, image, grade, concepts in tqdm(data_loader,
                                          desc="Validation CBM", unit="batch"):

        image = image.to(device, non_blocking=True)
        grade = grade.argmax(dim=1).long().to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(image, return_maps=False)
            grade_loss = grade_loss_fn(outputs["grade_logits"], grade)
            concept_loss = concept_loss_fn(outputs["concept_logits"], concepts)
            loss = (grade_loss_weight * grade_loss
                    + concept_loss_weight * concept_loss)

        batch_size = image.size(0)
        running_loss += loss.item() * batch_size
        running_grade_loss += grade_loss.item() * batch_size
        running_concept_loss += concept_loss.item() * batch_size
        number_of_images += batch_size

        targets.extend(grade.cpu().tolist())
        predictions.extend(outputs["grade_logits"].argmax(dim=1).cpu().tolist())
        concept_targets.extend(concepts.cpu().tolist())
        concept_predictions.extend((outputs["concept_logits"].float()
                                    .sigmoid() >= 0.5).int().cpu().tolist())

    metrics = calculate_grading_metrics(targets, predictions, num_grade_classes)
    metrics.update(calculate_concept_metrics(concept_targets, concept_predictions))
    metrics.update({"loss": running_loss / number_of_images,
                    "grade_loss": running_grade_loss / number_of_images,
                    "concept_loss": running_concept_loss / number_of_images})
    return metrics


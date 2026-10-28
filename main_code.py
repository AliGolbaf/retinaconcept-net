#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Written by: Ali Golbaf

    Task:
        Black-box and concept bottleneck models for diabetic retinopathy grading.

    Dataset:
        MMRDR-CFP (five DR grades and seven retinal lesion concepts).
"""

################################################################################
################################################################################
""" Libraries """
import os
import random
import json
from pathlib import Path

# Parse arguments first so --help and --dry-run do not require PyTorch.
from cli_config import parse_arguments
args = parse_arguments()

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data_loader import SubjectsDataset
from models import BlackBox_Model, Concept_Bottleneck_Model
from train import (train_one_epoch_blackbox,
                   validate_one_epoch_blackbox,
                   train_one_epoch_cbm,
                   validate_one_epoch_cbm)

from utils import (makedir, 
                   read_csv_exl, 
                   load_data,
                   NUM_DR_GRADES, 
                   NUM_DR_CONCEPTS, 
                   count_parameters)
from reporting import save_results

################################################################################
################################################################################
""" Experiment configuration """

mode = args.mode  # "Train_Black-box", "Train_CBM", "Test_Black-box" or "Test_CBM"

random_seed = args.seed
num_folds = args.num_folds
batch_size = args.batch_size
image_size = tuple(args.image_size)
num_workers = args.num_workers

number_of_epochs = args.epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
pretrained = args.pretrained
use_class_weights = args.class_weights
load_existing_models = args.load_existing_models
blackbox_patience = args.blackbox_patience

# CBM training ---------------------------------------------------------------
concept_epochs = args.concept_epochs
grading_epochs = args.grading_epochs
concept_patience = args.concept_patience
grading_patience = args.grading_patience

# Reproducibility -------------------------------------------------------------
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
if args.deterministic:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# CUDA ------------------------------------------------------------------------
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")
device = torch.device("cuda" if (args.device == "cuda" or
                      (args.device == "auto" and torch.cuda.is_available())) else "cpu")
print(f"Device: {device}")

################################################################################
################################################################################
""" Paths """

path_to_data = str(args.data_root)
path_to_data_images = os.path.join(path_to_data, args.images_subdir)

if mode in ("Train_CBM", "Test_CBM"):
    path_to_outputs = os.path.join(args.output_dir, "cbm_resnet152")
else:
    path_to_outputs = os.path.join(args.output_dir, "blackbox_resnet152")

makedir(path_to_outputs)

# Save the exact settings used for this run.
configuration = {name: str(value) if isinstance(value, Path) else value
                 for name, value in vars(args).items()}
configuration["torch_version"] = torch.__version__
with open(os.path.join(path_to_outputs, f"{mode.lower()}_configuration.json"),
          "w", encoding="utf-8") as output_file:
    json.dump(configuration, output_file, indent=2)

################################################################################
################################################################################
""" Load data """

data_clinical = read_csv_exl(path_to_data, args.annotations_file)
required_columns = {"image", "grade", "lesion"}
if not required_columns.issubset(data_clinical.columns):
    raise ValueError(f"Annotation file must contain {sorted(required_columns)}")
data = load_data(path_to_data_images,
                 data_clinical["image"].values,
                 data_clinical["grade"].values,
                 data_clinical["lesion"].values)

data_train = [subject for subject in data
              if subject["subject_id"].startswith("tr")]
data_test = [subject for subject in data
             if subject["subject_id"].startswith("ts")]

if not data_train or not data_test:
    raise ValueError("MMRDR-CFP must contain both tr* and ts* image records.")

grades_train = np.asarray([subject["grade"] for subject in data_train],
                          dtype=np.int64)

print(f"Training images: {len(data_train)}")
print(f"Test images: {len(data_test)}")

################################################################################
################################################################################
""" Five-fold cross-validation """

if mode in ("Train_Black-box", "Train_CBM"):
    cv = StratifiedKFold(n_splits=num_folds,
                         shuffle=True,
                         random_state=random_seed)
    cv_splits = cv.split(data_train, grades_train)
else:
    cv_splits = []

fold_results = []
fold_details = []

for fold_index, (train_index, validation_index) in enumerate(cv_splits, start=1):
    if args.fold is not None and fold_index != args.fold:
        continue

    # Keep the training and validation subjects independent.
    assert set(train_index).isdisjoint(validation_index)

    print("\n" + "=" * 70)
    print(f"{mode} | Fold {fold_index}/{num_folds}")
    print("=" * 70)

    path_to_fold = os.path.join(path_to_outputs, f"fold_{fold_index}")
    makedir(path_to_fold)

    # Training and validation data --------------------------------------------
    train_data = [data_train[index] for index in train_index]
    validation_data = [data_train[index] for index in validation_index]

    train_loader = DataLoader(SubjectsDataset(data=train_data,
                                              num_grade_classes=NUM_DR_GRADES,
                                              image_size=image_size,
                                              augment=True),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    validation_loader = DataLoader(SubjectsDataset(data=validation_data,
                                                   num_grade_classes=NUM_DR_GRADES,
                                                   image_size=image_size,
                                                   augment=False),
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers)

    # Grading class weights ---------------------------------------------------
    if use_class_weights:
        counts = np.bincount(grades_train[train_index],
                             minlength=NUM_DR_GRADES)
        if np.any(counts == 0):
            raise ValueError("A training fold is missing a grading class")
        class_weights = torch.tensor(counts.sum() / (NUM_DR_GRADES * counts),
                                     dtype=torch.float32,
                                     device=device)
    else:
        class_weights = None

    grade_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ############################################################################
    """ Black-box model """

    if mode == "Train_Black-box":

        checkpoint_path = os.path.join(path_to_fold, "best_model.pt")

        model = BlackBox_Model(num_grade_classes=NUM_DR_GRADES,
                               pretrained=pretrained).to(device)

        optimiser = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, 
                                                               T_max=number_of_epochs)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        if load_existing_models and os.path.isfile(checkpoint_path):

            model.load_state_dict(torch.load(checkpoint_path,
                                             map_location="cpu",
                                             weights_only=True))
            
            print(f"Loaded existing black-box: {checkpoint_path}")

            previous_metrics = validate_one_epoch_blackbox(model, 
                                                           validation_loader, 
                                                           grade_loss_fn, 
                                                           device, 
                                                           use_amp=use_amp)
            best_val_loss = previous_metrics["loss"]

        total, trainable, frozen = count_parameters(model)
        print(f"Parameters: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")

        # Training -------------------------------------------------------------
        for epoch in range(number_of_epochs):

            train_metrics = train_one_epoch_blackbox(model, 
                                                     train_loader, 
                                                     optimiser, 
                                                     grade_loss_fn,
                                                     device, 
                                                     scaler=scaler, 
                                                     use_amp=use_amp)

            val_metrics = validate_one_epoch_blackbox(model, 
                                                      validation_loader, 
                                                      grade_loss_fn,
                                                      device, 
                                                      use_amp=use_amp)

            scheduler.step()

            print(f"Fold {fold_index} | Epoch {epoch + 1:03d}/{number_of_epochs} | "
                  f"Train loss: {train_metrics['loss']:.4f} | "
                  f"Val loss: {val_metrics['loss']:.4f} | "
                  f"Train BAC: {train_metrics['balanced_accuracy']:.4f} | "
                  f"Val BAC: {val_metrics['balanced_accuracy']:.4f}")

            if val_metrics["loss"] < best_val_loss - 1e-5:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= blackbox_patience:
                print(f"Early stopping at epoch {epoch + 1}: "
                      f"no validation loss improvement for "
                      f"{blackbox_patience} epochs.")
                break            

        # Evaluate the best checkpoint ----------------------------------------
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location="cpu",
                                         weights_only=True))
        final_metrics = validate_one_epoch_blackbox(model, 
                                                    validation_loader, 
                                                    grade_loss_fn, 
                                                    device, 
                                                    use_amp=use_amp)

        fold_results.append({"accuracy": final_metrics["accuracy"],
                             "balanced_accuracy": final_metrics["balanced_accuracy"],
                             "macro_f1": final_metrics["macro_f1"],
                             "quadratic_kappa": final_metrics["quadratic_kappa"]})

    ############################################################################
    """ Concept Bottleneck Model """
    
    if mode == "Train_CBM":

        checkpoint_path = os.path.join(path_to_fold, "best_cbm_model.pt")
        concept_checkpoint_path = os.path.join(path_to_fold,
                                               "best_concept_model.pt")

        model = Concept_Bottleneck_Model(num_grade_classes=NUM_DR_GRADES,
                                         num_concepts=NUM_DR_CONCEPTS,
                                         pretrained=False).to(device)

        # Concept class weights --------------------------------------------------
        train_concept_labels = np.asarray(
            [subject["concepts"] for subject in train_data], dtype=np.float32)

        if use_class_weights:
            positive_counts = train_concept_labels.sum(axis=0)
            negative_counts = len(train_concept_labels) - positive_counts

            concept_weights = np.ones(NUM_DR_CONCEPTS, dtype=np.float32)
            valid_concepts = (positive_counts > 0) & (negative_counts > 0)

            concept_weights[valid_concepts] = (negative_counts[valid_concepts] / 
                                               positive_counts[valid_concepts])

            concept_weights = torch.tensor(concept_weights,
                                           dtype=torch.float32,
                                           device=device)
        else:
            concept_weights = None

        concept_loss_fn = nn.BCEWithLogitsLoss(pos_weight=concept_weights)

        ########################################################################
        """ Stage 1: Concept learning """

        if load_existing_models and os.path.isfile(concept_checkpoint_path):

            model.load_state_dict(torch.load(concept_checkpoint_path,
                                             map_location="cpu",
                                             weights_only=True))

            print(f"Loaded concept model: {concept_checkpoint_path}")

        else:

            # Initialise the backbone --------------------------------------------
            blackbox_checkpoint_path = os.path.join(args.output_dir,
                                                    "blackbox_resnet152",
                                                    f"fold_{fold_index}", 
                                                    "best_model.pt")

            if not os.path.isfile(blackbox_checkpoint_path):
                raise FileNotFoundError(
                    f"Train the black-box model for fold {fold_index} first: "
                    f"{blackbox_checkpoint_path}")

            blackbox_weights = torch.load(blackbox_checkpoint_path,
                                          map_location="cpu",
                                          weights_only=True)

            backbone_weights = {name[len("backbone."):]: weights
                for name, weights in blackbox_weights.items()
                if name.startswith("backbone.")}

            model.backbone.load_state_dict(backbone_weights)
            print(f"Loaded black-box backbone: {blackbox_checkpoint_path}")

            # Optimiser -----------------------------------------------------------
            concept_parameters = [parameter for name, parameter in model.named_parameters()
                if not name.startswith(("backbone.", "grade_head."))]

            optimiser = torch.optim.AdamW([{"params": model.backbone.parameters(),
                                            "lr": learning_rate * 0.1},
                                           {"params": concept_parameters,
                                            "lr": learning_rate}], 
                                          weight_decay=weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, 
                                                                   T_max=concept_epochs)

            best_concept_loss = float("inf")
            epochs_without_improvement = 0

            # Concept training ----------------------------------------------------
            for epoch in range(concept_epochs):

                train_metrics = train_one_epoch_cbm(model, 
                                                    train_loader, 
                                                    optimiser,
                                                    grade_loss_fn, 
                                                    concept_loss_fn, device,
                                                    grade_loss_weight=0.0,
                                                    concept_loss_weight=1.0,
                                                    scaler=scaler, use_amp=use_amp)

                val_metrics = validate_one_epoch_cbm(model, 
                                                     validation_loader,
                                                     grade_loss_fn, 
                                                     concept_loss_fn, 
                                                     device,
                                                     grade_loss_weight=0.0,
                                                     concept_loss_weight=1.0,
                                                     use_amp=use_amp)

                scheduler.step()

                print(f"Fold {fold_index} | Concept epoch "
                      f"{epoch + 1:03d}/{concept_epochs} | "
                      f"Train loss: {train_metrics['loss']:.4f} | "
                      f"Val loss: {val_metrics['loss']:.4f} | "
                      f"Concept BAC: {val_metrics['concept_balanced_accuracy']:.4f}")

                if val_metrics["concept_loss"] < best_concept_loss - 1e-5:
                    best_concept_loss = val_metrics["concept_loss"]
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), concept_checkpoint_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= concept_patience:
                    print(f"Concept early stopping at epoch {epoch + 1}")
                    break

        # Load the best concept model --------------------------------------------
        model.load_state_dict(torch.load(concept_checkpoint_path,
                                         map_location="cpu",
                                         weights_only=True))

        concept_metrics = validate_one_epoch_cbm(model, validation_loader,
                                                 grade_loss_fn, 
                                                 concept_loss_fn, 
                                                 device,
                                                 grade_loss_weight=0.0,
                                                 concept_loss_weight=1.0,
                                                 use_amp=use_amp)

        print(f"Best concept BAC: "
              f"{concept_metrics['concept_balanced_accuracy']:.4f}")

        ########################################################################
        """ Stage 2: DR grading """

        # Freeze the concept extractor ------------------------------------------
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("grade_head."))

        total, trainable, frozen = count_parameters(model)
        print(f"Parameters: {total:,} | Trainable: {trainable:,} | "
              f"Frozen: {frozen:,}")

        # Optimise only the grading head ----------------------------------------
        optimiser = torch.optim.AdamW(model.grade_head.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, 
                                                               T_max=grading_epochs)

        best_grading_loss = float("inf")
        epochs_without_improvement = 0

        # Grading training -------------------------------------------------------
        for epoch in range(grading_epochs):

            train_metrics = train_one_epoch_cbm(model, 
                                                train_loader, 
                                                optimiser,
                                                grade_loss_fn, 
                                                concept_loss_fn, 
                                                device,
                                                grade_loss_weight=1.0,
                                                concept_loss_weight=0.0,
                                                scaler=scaler, use_amp=use_amp)

            val_metrics = validate_one_epoch_cbm(model, 
                                                 validation_loader,
                                                 grade_loss_fn, 
                                                 concept_loss_fn, 
                                                 device,
                                                 grade_loss_weight=1.0,
                                                 concept_loss_weight=0.0,
                                                 use_amp=use_amp)

            scheduler.step()

            print(f"Fold {fold_index} | Grading epoch "
                  f"{epoch + 1:03d}/{grading_epochs} | "
                  f"Train loss: {train_metrics['loss']:.4f} | "
                  f"Val loss: {val_metrics['loss']:.4f} | "
                  f"Grade BAC: {val_metrics['balanced_accuracy']:.4f} | "
                  f"Concept BAC: {val_metrics['concept_balanced_accuracy']:.4f}")

            if val_metrics["grade_loss"] < best_grading_loss - 1e-5:
                best_grading_loss = val_metrics["grade_loss"]
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= grading_patience:
                print(f"Grading early stopping at epoch {epoch + 1}")
                break

        # Evaluate the best CBM --------------------------------------------------
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location="cpu",
                                         weights_only=True))

        final_metrics = validate_one_epoch_cbm(model, 
                                               validation_loader,
                                               grade_loss_fn, 
                                               concept_loss_fn, 
                                               device,
                                               grade_loss_weight=1.0,
                                               concept_loss_weight=0.0,
                                               use_amp=use_amp)

        fold_results.append({"accuracy": final_metrics["accuracy"],
                             "balanced_accuracy": final_metrics["balanced_accuracy"],
                             "macro_f1": final_metrics["macro_f1"],
                             "quadratic_kappa": final_metrics["quadratic_kappa"],
                             "concept_bac_best": concept_metrics["concept_balanced_accuracy"],
                             "concept_bac_final": final_metrics["concept_balanced_accuracy"]})    


    # Fold summary ------------------------------------------------------------
    print(f"\nFold {fold_index} final results:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f} | "
          f"BAC: {final_metrics['balanced_accuracy']:.4f} | "
          f"Macro-F1: {final_metrics['macro_f1']:.4f} | "
          f"QWK: {final_metrics['quadratic_kappa']:.4f}")

    if mode == "Train_CBM":
        print(f"Best concept BAC: {concept_metrics['concept_balanced_accuracy']:.4f} | "
              f"Final concept BAC: {final_metrics['concept_balanced_accuracy']:.4f}")

    print("Confusion matrix:")
    print(final_metrics["confusion_matrix"])
    fold_details.append({"fold": fold_index, "metrics": final_metrics})

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


################################################################################
################################################################################
""" Cross-validation summary """

if mode in ("Train_Black-box", "Train_CBM"):
    print("\nCross-validation results (mean +/- standard deviation):")

    for metric_name in fold_results[0]:
        values = np.asarray([result[metric_name] for result in fold_results],
                            dtype=np.float64)
        print(f"{metric_name}: {values.mean():.4f} +/- "
              f"{values.std(ddof=1 if len(values) > 1 else 0):.4f}")

    save_results(path_to_outputs, mode, configuration, fold_details)


################################################################################
################################################################################
""" Test the model """

if mode in ("Test_Black-box", "Test_CBM"):

    test_loader = DataLoader(SubjectsDataset(data=data_test,
                                             num_grade_classes=NUM_DR_GRADES,
                                             image_size=image_size,
                                             augment=False),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    grade_loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    test_results = []
    test_details = []

    for fold_index in range(1, num_folds + 1):
        if args.fold is not None and fold_index != args.fold:
            continue

        print(f"\n{mode} | Fold {fold_index}/{num_folds}")
        path_to_fold = os.path.join(path_to_outputs, f"fold_{fold_index}")

        if mode == "Test_CBM":
            model = Concept_Bottleneck_Model(num_grade_classes=NUM_DR_GRADES,
                                             num_concepts=NUM_DR_CONCEPTS,
                                             pretrained=False).to(device)
            checkpoint_path = os.path.join(path_to_fold, "best_cbm_model.pt")
        else:
            model = BlackBox_Model(num_grade_classes=NUM_DR_GRADES,
                                   pretrained=False).to(device)
            checkpoint_path = os.path.join(path_to_fold, "best_model.pt")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location="cpu",
                                         weights_only=True))

        if mode == "Test_CBM":
            test_metrics = validate_one_epoch_cbm(model, test_loader,
                                                  grade_loss_fn,
                                                  nn.BCEWithLogitsLoss(),
                                                  device,
                                                  grade_loss_weight=1.0,
                                                  concept_loss_weight=0.0,
                                                  use_amp=use_amp)
        else:
            test_metrics = validate_one_epoch_blackbox(model, 
                                                       test_loader,
                                                       grade_loss_fn, device,
                                                       use_amp=use_amp)

        test_results.append({"accuracy": test_metrics["accuracy"],
                             "balanced_accuracy": test_metrics["balanced_accuracy"],
                             "macro_f1": test_metrics["macro_f1"],
                             "quadratic_kappa": test_metrics["quadratic_kappa"]})

        if mode == "Test_CBM":
            test_results[-1]["concept_balanced_accuracy"] = test_metrics["concept_balanced_accuracy"]
        test_details.append({"fold": fold_index, "metrics": test_metrics})

        print(f"Test loss: {test_metrics['loss']:.4f} | "
              f"Accuracy: {test_metrics['accuracy']:.4f} | "
              f"BAC: {test_metrics['balanced_accuracy']:.4f} | "
              f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
              f"QWK: {test_metrics['quadratic_kappa']:.4f}")

        if mode == "Test_CBM":
            print(f"Concept BAC: {test_metrics['concept_balanced_accuracy']:.4f}")

        print("Confusion matrix:")
        print(test_metrics["confusion_matrix"])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nIndependent test results across fold models (mean +/- standard deviation):")

    for metric_name in test_results[0]:
        values = np.asarray([result[metric_name] for result in test_results],
                            dtype=np.float64)
        print(f"{metric_name}: {values.mean():.4f} +/- "
              f"{values.std(ddof=1 if len(values) > 1 else 0):.4f}")

    save_results(path_to_outputs, mode, configuration, test_details)

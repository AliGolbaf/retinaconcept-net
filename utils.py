#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Written by: Ali Golbaf (ali.golbaf71@gmail.com)

"""
################################################################################
################################################################################
""" Libraries """
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from functools import partial
import ast
import torch


################################################################################
################################################################################
""" Make Directory """
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

################################################################################
################################################################################
""" Data """
# Read csv / excal file
def read_csv_exl(path, name):
    root, ext = os.path.splitext(name)
    if ext == ".csv":
        file = pd.read_csv(os.path.join(path, name))
    if ext == ".xlsx":
        file = pd.read_excel(os.path.join(path, name))
    return file  

################################################################################
################################################################################
""" Load Fundus Data """
def load_data(path_to_data_images, subjects, grades, concepts):
    """
    Organise fundus image paths, grading labels and lesion concepts.

    Args:
        path_to_data_images: Directory containing the fundus JPG images.
        subjects: Image names from FP.csv, e.g. "img/tr000001.jpg".
        grades: Diabetic retinopathy grading labels.
        concepts: Seven binary lesion indicators for each image.

    Returns:
        data: List of subjects with image paths, grades and concepts.
    """
    if not (len(subjects) == len(grades) == len(concepts)):
        raise ValueError("subjects, grades and concepts must have equal lengths")

    data = []

    for subject, grade, concept in zip(subjects, grades, concepts):
        # FP.csv paths begin with 'img/', but path_to_data_images already points to img.
        image_path = os.path.join(path_to_data_images, os.path.basename(str(subject)))

        # Accept either parsed concept arrays or strings from FP.csv.
        if isinstance(concept, str):
            concept = ast.literal_eval(concept)

        if len(concept) != NUM_DR_CONCEPTS:
            raise ValueError(f"Expected {NUM_DR_CONCEPTS} concepts for {subject}")

        data.append({"subject_id": os.path.basename(str(subject)),
                     "image_path": image_path,
                     "grade": int(grade),
                     "concepts": [int(value) for value in concept]})

    return data


################################################################################
################################################################################
""" Retinopathy Grades """

# Diabetic retinopathy grading labels
DR_GRADES = {0: "No DR",
             1: "Mild NPDR",
             2: "Moderate NPDR",
             3: "Severe NPDR",
             4: "Proliferative DR"}

NUM_DR_GRADES = len(DR_GRADES)

################################################################################
################################################################################
""" Retinopathy Concepts """

# Lesion labels in the same order as FP.csv
DR_CONCEPTS = {0: "Microaneurysms (MA)",
               1: "Hard Exudates (HE)",
               2: "Intraretinal Haemorrhages (IH)",
               3: "Venous Beading / IRMA (VB/IRMA)",
               4: "Neovascularisation (NV)",
               5: "Vitreous Haemorrhage (VH)",
               6: "Retinal Detachment (RD)"}

NUM_DR_CONCEPTS = len(DR_CONCEPTS)


################################################################################
################################################################################
""" Model Parameters """

def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    frozen_params = total_params - trainable_params

    return total_params, trainable_params, frozen_params
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Written by: Ali Golbaf (ali.golbaf71@gmail.com)

"""
################################################################################
################################################################################
""" Libraries """
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


################################################################################
################################################################################
""" Classes """

class SubjectsDataset(Dataset):

    def __init__(self,
                 data,
                 num_grade_classes=5,
                 image_size=(224, 224),
                 augment=False):

        self.data = data
        self.num_grade_classes = num_grade_classes
        self.augment = augment

        # Image preprocessing ---------------------------------------------------
        transform_sequence = [transforms.Resize(image_size)]

        # Image augmentation (training only) -----------------------------------
        if self.augment:
            transform_sequence.extend([transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation(degrees=10),
                                       transforms.ColorJitter(brightness=0.1,
                                                              contrast=0.1,
                                                              saturation=0.1)])
        
        # Tensor conversion and normalisation ----------------------------------
        transform_sequence.append(transforms.ToTensor())

        transform_sequence.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_sequence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        subject = self.data[index]

        # Subject ID
        subject_id = subject["subject_id"]

        # Fundus image: RGB [3, H, W]
        with Image.open(subject["image_path"]) as image_file:
            image = image_file.convert("RGB")
            image = self.transform(image)

        # Diabetic retinopathy grade
        grade = torch.tensor(subject["grade"], dtype=torch.long)
        grade = torch.nn.functional.one_hot(grade,
                                            num_classes=self.num_grade_classes)

        # Seven binary lesion concepts
        concepts = torch.tensor(subject["concepts"], dtype=torch.float32)

        return subject_id, image, grade, concepts
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Written by: Ali Golbaf (ali.golbaf71@gmail.com)

"""

################################################################################
################################################################################
""" Libraries """
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet152, ResNet152_Weights


################################################################################
################################################################################
""" Classes """

class ResNet_152_Backbone(nn.Module):
    """
    Image input: [B, 3, H, W].

    Returns:
        feature_2: [B, 512,  H/8,  W/8]
        feature_3: [B, 1024, H/16, W/16]
        feature_4: [B, 2048, H/32, W/32]
    """

    def __init__(self,
                 pretrained=True):
        super().__init__()

        # Torchvision architecture and optional ImageNet pretrained weights
        weights = ResNet152_Weights.DEFAULT if pretrained else None
        network = resnet152(weights=weights)

        # Preserve access to intermediate spatial representations
        self.stem = nn.Sequential(network.conv1,
                                  network.bn1,
                                  network.relu,
                                  network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4
        self.avgpool = network.avgpool

    def forward(self, x):

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected RGB input [B, 3, H, W]")

        x = self.stem(x)
        x = self.layer1(x)

        feature_2 = self.layer2(x)
        feature_3 = self.layer3(feature_2)
        feature_4 = self.layer4(feature_3)

        # Global feature vector for the black-box grading head
        global_features = torch.flatten(self.avgpool(feature_4), 1)

        return {"feature_2": feature_2,
                "feature_3": feature_3,
                "feature_4": feature_4,
                "global_features": global_features}


################################################################################
################################################################################
""" Black-box model """

class BlackBox_Model(nn.Module):
    """Direct prediction of DR grades from deep ResNet-152 features."""

    def __init__(self,
                 num_grade_classes=5,
                 pretrained=True,
                 dropout=0.2):
        super().__init__()

        self.backbone = ResNet_152_Backbone(pretrained=pretrained)
        self.grade_head = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(2048, num_grade_classes))

    def forward(self, x):

        backbone_outputs = self.backbone(x)
        grade_logits = self.grade_head(backbone_outputs["global_features"])

        return {"grade_logits": grade_logits}


################################################################################
################################################################################
""" Concept Bottleneck Model """

class Concept_Bottleneck_Model(nn.Module):
    """
    Predicts seven lesion concepts using multi-scale spatial features.
    The grading head receives ONLY the seven concept probabilities.

    Returns:
        grade_logits: [B, num_grade_classes]
        concept_logits: [B, num_concepts]
        concept_probs: [B, num_concepts]
        concept_maps: [B, num_concepts, H/8, W/8] (if return_maps=True)

    Args:
        topk_fraction: Fraction of strongest spatial responses used to
                       aggregate each concept map into its image-level logit.
    """

    def __init__(self,
                 num_grade_classes=5,
                 num_concepts=7,
                 pretrained=True,
                 concept_channels=128,
                 topk_fraction=0.05):
        super().__init__()

        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")

        self.backbone = ResNet_152_Backbone(pretrained=pretrained)
        self.num_concepts = num_concepts
        self.topk_fraction = topk_fraction

        # Multi-scale feature projection ---------------------------------------
        self.project_2 = nn.Conv2d(512, concept_channels, kernel_size=1)
        self.project_3 = nn.Conv2d(1024, concept_channels, kernel_size=1)
        self.project_4 = nn.Conv2d(2048, concept_channels, kernel_size=1)

        # Spatial concept prediction head -------------------------------------
        self.concept_head = nn.Sequential(
            nn.Conv2d(3 * concept_channels,
                      concept_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(concept_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(concept_channels, num_concepts, kernel_size=1))

        # Strict concept bottleneck: no direct image-feature bypass ------------
        self.grade_head = nn.Linear(num_concepts, num_grade_classes)

    def forward(self, x, return_maps=True):

        backbone_outputs = self.backbone(x)

        feature_2 = backbone_outputs["feature_2"]
        feature_3 = backbone_outputs["feature_3"]
        feature_4 = backbone_outputs["feature_4"]

        # Align deeper spatial features with layer2 resolution
        output_size = feature_2.shape[-2:]

        feature_2 = self.project_2(feature_2)
        feature_3 = F.interpolate(self.project_3(feature_3),
                                  size=output_size,
                                  mode="bilinear",
                                  align_corners=False)
        feature_4 = F.interpolate(self.project_4(feature_4),
                                  size=output_size,
                                  mode="bilinear",
                                  align_corners=False)

        shared_features = torch.cat([feature_2, feature_3, feature_4], dim=1)

        # Concept evidence maps: raw logits, then sigmoid for visualisation
        concept_map_logits = self.concept_head(shared_features)

        # Top-k spatial aggregation: image-level concept logits
        flat_maps = concept_map_logits.flatten(start_dim=2)
        num_positions = flat_maps.shape[-1]
        num_selected = max(1, int(round(num_positions * self.topk_fraction)))

        concept_logits = torch.topk(flat_maps,
                                    k=num_selected,
                                    dim=2).values.mean(dim=2)
        concept_probs = torch.sigmoid(concept_logits)

        # Grade predictions depend ONLY on concept probabilities
        grade_logits = self.grade_head(concept_probs)

        outputs = {"grade_logits": grade_logits,
                   "concept_logits": concept_logits,
                   "concept_probs": concept_probs}

        if return_maps:
            outputs["concept_maps"] = torch.sigmoid(concept_map_logits)

        return outputs
    
    
    
################################################################################
################################################################################
""" Model check """

if __name__ == "__main__":

    # Generate a single random RGB image (this is not a retinal photograph).
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(1, 3, 224, 224, device=device)

    print(f"Device: {device}")
    print(f"Input image: {tuple(image.shape)}")

    # Check the black-box model -----------------------------------------------
    blackbox = BlackBox_Model(num_grade_classes=5,
                              pretrained=False).to(device)
    blackbox.eval()

    with torch.no_grad():
        outputs = blackbox(image)
        grade_probs = torch.softmax(outputs["grade_logits"], dim=1)

    print("\nBlack-box model:")
    print(f"Grade logits: {tuple(outputs['grade_logits'].shape)}")
    print(f"Grade probabilities: {grade_probs[0].cpu().tolist()}")
    print(f"Predicted grade: {grade_probs.argmax(dim=1).item()}")

    del blackbox

    # Check the concept bottleneck model --------------------------------------
    cbm = Concept_Bottleneck_Model(num_grade_classes=5,
                                   num_concepts=7,
                                   pretrained=False).to(device)
    cbm.eval()

    with torch.no_grad():
        outputs = cbm(image, return_maps=True)
        grade_probs = torch.softmax(outputs["grade_logits"], dim=1)

    print("\nConcept Bottleneck Model:")
    print(f"Grade logits: {tuple(outputs['grade_logits'].shape)}")
    print(f"Concept logits: {tuple(outputs['concept_logits'].shape)}")
    print(f"Concept maps: {tuple(outputs['concept_maps'].shape)}")
    print(f"Grade probabilities: {grade_probs[0].cpu().tolist()}")
    print(f"Concept probabilities: {outputs['concept_probs'][0].cpu().tolist()}")
    print(f"Predicted grade: {grade_probs.argmax(dim=1).item()}")

    print("\nForward-pass check completed successfully.")

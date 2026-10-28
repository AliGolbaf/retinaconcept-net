#!/usr/bin/env python3
"""
    Research project: Ali Golbaf

    Task:
        Generate concept evidence overlays from a trained CBM checkpoint.

    Note:
        Evidence activations are NOT validated lesion segmentation masks.
"""

from __future__ import annotations

################################################################################
################################################################################
""" Libraries """

import argparse
import json
from pathlib import Path


GRADE_NAMES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")
CONCEPT_CODES = ("MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD")
CONCEPT_NAMES = (
    "Microaneurysms", "Hard exudates", "Intraretinal haemorrhages",
    "Venous beading / IRMA", "Neovascularisation", "Vitreous haemorrhage",
    "Retinal detachment",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="A trained best_cbm_model.pt checkpoint.")
    parser.add_argument("--image", type=Path, required=True,
                        help="A colour fundus JPG/PNG supplied by the authorised dataset holder.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explanations"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--concept", choices=("all", *map(str, range(7))), default="all",
                        help="Which concept maps to render (0..6 or all).")
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-display-size", type=int, default=1200,
                        help="Limit width/height of the saved fundus image, in pixels.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.opacity <= 1 or not 0 <= args.threshold < 1:
        raise SystemExit("--opacity must be in [0,1] and --threshold in [0,1)")
    if args.max_display_size < 128 or min(args.image_size) < 32:
        raise SystemExit("Display size must be >= 128 and input dimensions >= 32")
    if not args.checkpoint.is_file() or not args.image.is_file():
        raise SystemExit("Provide existing --checkpoint and --image files")

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torchvision import transforms

    from models import Concept_Bottleneck_Model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = Concept_Bottleneck_Model(num_grade_classes=5, num_concepts=7,
                                    pretrained=False).to(device)
    # weights_only=True prevents loading arbitrary Python objects from a checkpoint.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with Image.open(args.image) as source:
        rgb = source.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(tuple(args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        result = model(transform(rgb).unsqueeze(0).to(device), return_maps=True)
        grade_scores = torch.softmax(result["grade_logits"], dim=1)[0].float().cpu().tolist()
        concept_scores = result["concept_probs"][0].float().cpu().tolist()
        maps = result["concept_maps"][0].float().cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.image.stem
    prediction = int(np.argmax(grade_scores))
    summary = {
        "source_image": args.image.name,
        "checkpoint": args.checkpoint.name,
        "predicted_grade": prediction,
        "predicted_grade_label": GRADE_NAMES[prediction],
        "grade_softmax_scores_uncalibrated": dict(zip(GRADE_NAMES, grade_scores)),
        "concept_scores": dict(zip(CONCEPT_NAMES, concept_scores)),
        "warning": "Research-only predictions; concept evidence is not a lesion segmentation mask.",
    }
    (args.output_dir / f"{filename}_prediction.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    display = rgb.copy()
    display.thumbnail((args.max_display_size, args.max_display_size), Image.Resampling.LANCZOS)
    base = np.asarray(display, dtype=np.float32)
    indexes = range(7) if args.concept == "all" else [int(args.concept)]
    for index in indexes:
        map_image = Image.fromarray(np.uint8(np.clip(maps[index], 0, 1) * 255), "L")
        map_image = map_image.resize(display.size, Image.Resampling.BILINEAR)
        activation = np.asarray(map_image, dtype=np.float32) / 255
        response = np.clip((activation - args.threshold) / (1 - args.threshold), 0, 1)

        # Distinct cyan -> amber -> red colour response, as in the desktop explorer.
        heat = np.stack((np.clip(2.2 * response, 0, 1),
                         np.clip(1.2 - abs(response - 0.45) * 2, 0, 1),
                         np.clip(1.3 - 2.1 * response, 0, 1)), axis=-1) * 255
        alpha = (response * args.opacity)[..., None]
        overlay = Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + heat * alpha,
                                                   0, 255)), "RGB")

        # Save original and explanation side by side with a small descriptive header.
        canvas = Image.new("RGB", (display.width * 2, display.height + 48), "#102033")
        canvas.paste(display, (0, 48))
        canvas.paste(overlay, (display.width, 48))
        drawing = ImageDraw.Draw(canvas)
        drawing.text((12, 12), "Original CFP", fill="white")
        drawing.text((display.width + 12, 12),
                     f"{CONCEPT_CODES[index]} evidence | score {concept_scores[index]:.3f}",
                     fill="white")
        target = args.output_dir / f"{filename}_concept_{CONCEPT_CODES[index]}.png"
        canvas.save(target)
        print(f"Saved {target}")
    print(f"Predicted grade: {GRADE_NAMES[prediction]} (uncalibrated score {grade_scores[prediction]:.3f})")


if __name__ == "__main__":
    main()

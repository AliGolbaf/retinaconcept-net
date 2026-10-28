# RetinalConcept-Net

**Author: Ali Golbaf**

Research code for five-class diabetic retinopathy (DR) grading using colour fundus photographs (CFP) from MMRDR. This repository contains the original ResNet-152 black-box and Concept Bottleneck Model (CBM) implementations, with command-line configuration and reproducibility utilities.

The original model architecture, data loader, training functions and utility functions have been retained. Changes to `main_code.py` expose the experiment settings as arguments and save metrics. Additional scripts provide independent case-level evaluation and visualisation of the CBM's learned concept evidence. The graphical user interface is a **separate project** and is not included here.

> **Research use only.** No clinical validation is claimed. Softmax scores are not calibrated clinical probabilities. Learned concept maps should not be interpreted as confirmed lesion segmentation or clinical ground truth.

## 1. Repository files

| File | Description |
| --- | --- |
| `main_code.py` | Original training/testing experiment, now driven by command-line arguments |
| `models.py` | Original black-box and CBM architecture (unchanged) |
| `train.py` | Original training and validation functions (unchanged) |
| `data_loader.py` | Original image preprocessing and augmentation (unchanged) |
| `utils.py` | Original annotation loading and grade/concept labels (unchanged) |
| `cli_config.py` | Argument definitions and validation |
| `reporting.py` | JSON output and per-class DR grading metrics |
| `evaluate.py` | Independent test-set predictions, reliability and error analysis |
| `explain.py` | Case-level CBM concept activation visualisations |
| `tests/` | Dataset-free unit tests |

## 2. Dataset

The source dataset is described in:

Tang, Z. et al. (2026). *A multimodal retinal image dataset for diabetic retinopathy detection using foundation models*. Scientific Data, 13, 639. [DOI: 10.1038/s41597-026-07005-9](https://doi.org/10.1038/s41597-026-07005-9).

Data access: [MMRDR on Figshare](https://figshare.com/articles/dataset/MMRDR/29423747).

Download the **MMRDR-CFP** subset separately, subject to its access/licensing conditions. The dataset and associated annotations are **not** distributed in this repository.

The expected local layout is:

```text
MMRDR-CFP/
  FP.csv
  img/
    tr000001.jpg
    ...
    ts000001.jpg
    ...
```

`FP.csv` must contain `image`, `grade` (0–4) and `lesion` (seven binary concept labels). The original data loading code uses `tr*` for training/validation and `ts*` for the independent test set. The CFP split in the dataset paper is at **image level** because patient identifiers are unavailable; patient-level independence must not be claimed.

## 3. Installation

Python **3.11 or 3.12** is recommended. Create a separate environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Choose a PyTorch/torchvision build appropriate for your CPU or NVIDIA CUDA installation using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then install the other dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirement file records supported version ranges. For precise reproduction, record exact versions on the computer used for the experiment:

```bash
python -m pip freeze > local_environment_lock.txt
```

Retain that file with your experiment record; check it for private information before publishing it.

## 4. Running experiments

View all input arguments:

```bash
python main_code.py --help
```

All dataset and output paths are now supplied as arguments; no personal drive path is needed. You can first check your input layout without importing PyTorch:

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --dry-run
```

### Black-box model training

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --num-folds 5 --batch-size 32 --epochs 100 --learning-rate 0.0001 --seed 42
```

The best checkpoint from each fold is saved to `outputs/blackbox_resnet152/fold_N/best_model.pt`.

### Concept Bottleneck Model training

Run the black-box training stage first. The CBM initialises its backbone from the matching black-box fold and then learns seven lesion concepts before training the linear DR grading head.

```bash
python main_code.py --mode Train_CBM --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --num-folds 5 --concept-epochs 100 --grading-epochs 100 --seed 42
```

The best concept and complete CBM weights are saved in `outputs/cbm_resnet152/fold_N/`.

### Original experiment settings

The default values match the supplied experiment script: seed 42, five stratified folds, batch size 32, image size 224×224, AdamW learning rate 1e-4 and weight decay 1e-4. Black-box training uses ImageNet initialisation by default. The original two-stage CBM training procedure is unchanged.

You can change these settings with `--seed`, `--batch-size`, `--image-size`, `--num-workers`, `--device`, `--learning-rate`, `--weight-decay`, `--epochs`, `--concept-epochs`, `--grading-epochs` and early-stopping arguments. Use `--fold 1` to run one of the five fold splits for a pipeline check.

**Important:** `--load-existing-models` is enabled by default, preserving the original script's behaviour. To train without reusing existing model weights, pass `--no-load-existing-models` and preferably select a fresh `--output-dir`. Loading weights is **not** a full optimiser/scheduler resume.

### Test evaluation

```bash
python main_code.py --mode Test_Black-box --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs
python main_code.py --mode Test_CBM --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs
```

These commands assess the independently trained fold models one at a time on the `ts*` images; they do not construct an ensemble. The console and saved JSON files report accuracy, balanced accuracy, macro-F1, quadratic weighted kappa, per-class precision/recall/F1, the confusion matrix, and the number of grade underestimations. Configuration files and metrics JSON are saved under `outputs/`.

## 5. Case-level error and reliability analysis

For a detailed independent assessment using a selected checkpoint:

```bash
python evaluate.py --model cbm --checkpoint outputs/cbm_resnet152/fold_1/best_cbm_model.pt --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs/evaluation
```

Or specify `--model blackbox` and its matching checkpoint. The resulting JSON includes each test image ID, actual and predicted grades, uncalibrated softmax scores, underestimation indicators and (for CBM) predicted concept scores. It also reports top-class 10-bin expected calibration error (ECE) as a **descriptive** reliability measure; ECE does not calibrate a model. Use the case records to review representative classification errors, particularly severe underestimation and confidently incorrect predictions.

## 6. Concept evidence visualisations

```bash
python explain.py --checkpoint outputs/cbm_resnet152/fold_1/best_cbm_model.pt --image "D:/Datasets/MMRDR-CFP/img/ts000001.jpg" --output-dir outputs/explanations --concept all
```

This writes one JSON prediction summary and seven PNGs comparing the original fundus image with each concept's evidence overlay. Adjust `--opacity`, `--threshold`, `--concept` or `--device` when required. The maps are low-resolution learned responses, not lesion annotations or causal explanations.

To meet the assessment's example-output requirement, generate examples locally using authorised images. Only share them if dataset permissions allow it. Real experimental results have not been invented or included in this repository.

## 7. Reproducibility and limitations

1. Do not use the `ts*` test set for training, validation or hyperparameter selection.
2. Use the same random seed, fold count and output root when comparing the two models.
3. The source code seeds Python, NumPy and PyTorch. `--deterministic` requests deterministic PyTorch execution where supported; exact reproducibility can still depend on hardware and libraries.
4. Five-fold stratification is based on image-level DR grades. Without patient identifiers, cross-image patient leakage cannot be ruled out.
5. Record your actual command, package versions, checkpoint provenance and the generated metrics alongside the report.
6. The CBM's five-grade decision uses the seven predicted concept probabilities only; its spatial maps are exploratory model evidence.
7. Never upload raw CFP images, `FP.csv`, checkpoints, confidential metadata or automatically generated outputs to this public repository. `.gitignore` helps, but review files before pushing.

### Authorship and acknowledgements

The original black-box model, CBM model, training, data-loading and utility code were provided by **Ali Golbaf**. Repository command-line, evaluation and documentation utilities were prepared around that research code and should be reviewed by the project author before publication. Acknowledge any AI assistance in accordance with the applicable assessment policy. Cite the MMRDR dataset and relevant pretrained ResNet/torchvision resources.

No open-source licence has been selected; this requires the author's or PI's approval before publication.

## 8. Publishing to GitHub

Create a new, empty GitHub repository (for example, `RetinalConcept-Net`). Upload the **contents of this folder** to the repository root; do not add the separate GUI project. Keep the source code, README, dependency file, citation and tests visible. Before pushing, check the files staged for publication and ensure there are no images, datasets, checkpoints or private outputs. Confirm the assessor can access the repository link.

# RetinaConcept-Net

This repository contains the official implementation of **RetinaConcept-Net**, a concept-based deep learning framework for explainable diabetic retinopathy classification using colour fundus photographs, developed by **Ali Golbaf**. 

The proposed framework integrates a **ResNet-152** backbone with a **Concept Bottleneck Model (CBM)** to perform five-class diabetic retinopathy grading using seven retinal lesion concepts, providing interpretable evidence to support model predictions.
### 🔄 RetinaConcept-Net Workflow
![RetinaConcept-Net Workflow](workflow.png)

### 🧠 Core ideas
* **Interpretable grading through a Concept Bottleneck Model**  
  RetinaConcept-Net introduces an explicit concept bottleneck between retinal image features and diabetic retinopathy (DR) severity predictions. Instead of directly predicting DR grades from deep image features, the model first predicts seven retinal lesion concepts, including microaneurysms, hard exudates, intraretinal haemorrhages, and neovascularisation. These concept predictions are then used to classify fundus images into five DR severity grades. By restricting the grading head to concept-level information, the framework provides a more transparent representation of the evidence contributing to its predictions.

* **Multi-scale retinal feature extraction and concept localisation**  
  RetinaConcept-Net employs a ResNet-152 backbone to extract spatial features from three intermediate network stages. These representations are projected into a shared feature space, aligned to a common spatial resolution, and fused to capture retinal abnormalities at different scales. A dedicated concept prediction head generates spatial activation maps for each lesion concept, enabling visual examination of the retinal regions contributing to concept predictions. These maps represent learned model evidence rather than clinically validated lesion segmentations.

* **Two-stage training with concept-guided DR classification**  
  RetinaConcept-Net adopts a two-stage training strategy to separate concept learning from disease severity classification. In the first stage, the concept prediction network is trained using lesion-level annotations, with top-k spatial pooling aggregating the strongest activation responses into image-level concept predictions. In the second stage, the concept extractor is frozen, and the grading head is trained to predict DR severity exclusively from the seven concept probabilities. This design allows concept recognition and grading performance to be evaluated separately, while a conventional ResNet-152 black-box model provides a baseline for comparison.


### 🗂️ Repository files

| File | Description |
| --- | --- |
| `main_code.py` | Original training/testing experiment, now driven by command-line arguments |
| `models.py` | Original black-box and CBM architecture |
| `train.py` | Original training and validation functions |
| `data_loader.py` | Original image preprocessing and augmentation|
| `utils.py` | Original annotation loading and grade/concept labels |
| `cli_config.py` | Argument definitions and validation |
| `reporting.py` | JSON output and per-class DR grading metrics |
| `evaluate.py` | Independent test-set predictions, reliability and error analysis |
| `explain.py` | Case-level CBM concept activation visualisations |
| `tests/` | Dataset-free unit tests |

### 🖼️ Retinal Image Dataset

The source dataset is described in:
Tang, Z. et al. (2026). *A multimodal retinal image dataset for diabetic retinopathy detection using foundation models*. Scientific Data, 13, 639. [DOI: 10.1038/s41597-026-07005-9](https://doi.org/10.1038/s41597-026-07005-9).

Data access: [MMRDR on Figshare](https://figshare.com/articles/dataset/MMRDR/29423747).

### ⚙️ Installation
Python 3.11 or 3.12 is recommended.

**1. Clone the repository**

```bash
git clone https://github.com/AliGolbaf/retinaconcept-net.git
cd retinaconcept-net
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

First, install the appropriate CPU or CUDA version of PyTorch following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

**5. Verify the installation**

```bash
python main_code.py --help
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

### 🚀 Running experiments

View all input arguments:

```bash
python main_code.py --help
```

All dataset and output paths are now supplied as arguments; no personal drive path is needed. You can first check your input layout without importing PyTorch:

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --dry-run
```

#### Black-box model training

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --num-folds 5 --batch-size 32 --epochs 100 --learning-rate 0.0001 --seed 42
```

The best checkpoint from each fold is saved to `outputs/blackbox_resnet152/fold_N/best_model.pt`.

#### Concept Bottleneck Model training

Run the black-box training stage first. The CBM initialises its backbone from the matching black-box fold and then learns seven lesion concepts before training the linear DR grading head.

```bash
python main_code.py --mode Train_CBM --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --num-folds 5 --concept-epochs 100 --grading-epochs 100 --seed 42
```

The best concept and complete CBM weights are saved in `outputs/cbm_resnet152/fold_N/`.

#### Original experiment settings

The default values match the supplied experiment script: seed 42, five stratified folds, batch size 32, image size 224×224, AdamW learning rate 1e-4 and weight decay 1e-4. Black-box training uses ImageNet initialisation by default. The original two-stage CBM training procedure is unchanged.

You can change these settings with `--seed`, `--batch-size`, `--image-size`, `--num-workers`, `--device`, `--learning-rate`, `--weight-decay`, `--epochs`, `--concept-epochs`, `--grading-epochs` and early-stopping arguments. Use `--fold 1` to run one of the five fold splits for a pipeline check.

**Important:** `--load-existing-models` is enabled by default, preserving the original script's behaviour. To train without reusing existing model weights, pass `--no-load-existing-models` and preferably select a fresh `--output-dir`. Loading weights is **not** a full optimiser/scheduler resume.

#### Test evaluation

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

No open-source licence has been selected; this requires the author's or PI's approval before publication.

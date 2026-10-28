# RetinalConcept-Net

**Reproducible, concept-based diabetic retinopathy grading from colour fundus photographs.**

Research implementation by **Ali Golbaf**. This repository contains the experiment code used to train and evaluate a ResNet-152 black-box baseline and a strict concept bottleneck model (CBM) on the **MMRDR-CFP** dataset. The CBM estimates seven lesion-related concepts using multi-scale image features and predicts one of five diabetic retinopathy (DR) grades **only from its seven concept probabilities**.

> **Research use only.** This code is not a validated medical device. Softmax scores are not calibrated clinical probabilities. Concept activation maps are *model evidence*, not verified pixel-wise lesion segmentations. Human clinical judgement remains essential.

## Repository contents

| File | Purpose |
| --- | --- |
| `main_code.py` | Command-line interface (`--help`) for training and evaluation |
| `experiment.py` | The original two-stage CBM and black-box experimental procedure, parameterised |
| `models.py` | Original ResNet-152 architectures, unchanged in behaviour |
| `train.py` | Training/validation loops, per-class metrics and reliability summaries |
| `data_loader.py` | CFP image preprocessing and augmentation |
| `utils.py` | Annotation loading, grade/concept labels, parameter counts |
| `explain.py` | Single-image predictions and seven concept-evidence comparison PNGs |
| `requirements.txt` | Python dependency specifications |
| `tests/` | CLI checks runnable without dataset or checkpoint files |
| `.github/workflows/` | Automatic syntax and dataset-free CLI checks on GitHub |
| `CITATION.cff` | Software citation metadata |

No dataset images, clinical metadata, checkpoint weights or measured study results are included here.

## 1. Obtain and prepare the dataset

MMRDR dataset paper: [Tang et al., *Scientific Data* (2026)](https://doi.org/10.1038/s41597-026-07005-9). Dataset access: [MMRDR on Figshare](https://figshare.com/articles/dataset/MMRDR/29423747). Follow the dataset provider's conditions before downloading or processing the data.

Download and extract only the relevant **MMRDR-CFP** subset to a private local directory:

```text
MMRDR-CFP/
  FP.csv
  img/
    tr000001.jpg
    tr000002.jpg
    ...
    ts000001.jpg
    ...
```

`FP.csv` must have columns `image`, `grade`, and `lesion`. `grade` contains integers 0–4; `lesion` contains seven binary concept values in the dataset's published order. Filenames prefixed `tr` are used for development; `ts` images are reserved for independent testing. The loader handles the `img/...` prefixes in the CSV.

**Important data-splitting limitation:** The MMRDR dataset paper reports that its CFP train/test split is **at image level** because source patient identifiers were unavailable. Our five-fold stratification is also image based; subject/patient-level independence cannot be established from those identifiers alone. Do not claim patient-independent validation. Do not use test images to choose hyperparameters or checkpoints.

## 2. Installation

Python **3.11 or 3.12** recommended, with sufficient RAM/VRAM for ResNet-152 training.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install a compatible PyTorch + torchvision build for your CPU or GPU via the [official installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

See `requirements.txt` for the supported dependency version ranges. To record the **exact** versions used to reproduce results, run `python -m pip freeze > local_environment_lock.txt` after installation; retain that file with the experiment record (and publish it after PI review if appropriate). Version ranges are not a substitute for an archived, tested lockfile.

## 3. Command-line arguments

```bash
python main_code.py --help
```

The previous hard-coded `E:\...\MMRDR-CFP` location is removed. Use `--data-root` to point to your own authorised copy. Paths containing spaces should be quoted. The mode, seed, fold count/selection, batch size, image resolution, workers, epochs, learning rate, weight decay, early-stopping patience, ImageNet initialisation, class weights, device and output folder are all configurable.

Before using PyTorch, verify that the input locations resolve:

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --dry-run
```

### Train the black-box baseline

```bash
python main_code.py --mode Train_Black-box --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --seed 42 --folds 5 --batch-size 32 --epochs 100 --learning-rate 0.0001 --device auto
```

This saves one best-validation-loss checkpoint per fold at `outputs/blackbox_resnet152/fold_N/best_model.pt`.

### Train the concept bottleneck model

```bash
python main_code.py --mode Train_CBM --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs --seed 42 --folds 5 --concept-epochs 100 --grading-epochs 100 --device auto
```

**Run black-box training first.** CBM stage 1 initialises its backbone from the **matching black-box fold**. It trains spatial concept predictors and then freezes the concept extractor to train a linear grading head on predicted concept probabilities only. Outputs are written to `outputs/cbm_resnet152/fold_N/` as `best_concept_model.pt` and `best_cbm_model.pt`.

For a quick single-fold test (not a full reported experiment), append `--fold 1 --epochs 2` to black-box training or `--fold 1 --concept-epochs 2 --grading-epochs 2` to CBM training. These short runs are only pipeline checks.

**Checkpoint reuse:** `--reuse-checkpoints` initialises model weights from existing files. It **does not** restore optimiser, scheduler, scaler, or epoch state, so it is *not* a strict training resume. Without this flag, a training run starts a new fit and may replace checkpoints for the selected fold. Use a new `--output-dir` to preserve prior experiments.

## 4. Independent test evaluation

After training (and freezing hyperparameter choices):

```bash
python main_code.py --mode Test_Black-box --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs
python main_code.py --mode Test_CBM --data-root "D:/Datasets/MMRDR-CFP" --output-dir outputs
```

These commands evaluate each saved fold model **separately** on the `ts` images. They report the mean/standard deviation of the per-fold metrics; **they are not test-set ensemble predictions**. Use `--fold 1` to evaluate only one fold.

Training and evaluation automatically save machine-readable JSON under the matching model output directory:

- `train_black-box_metrics.json`, `train_cbm_metrics.json` — validation metrics per fold.
- `test_black-box_metrics.json`, `test_cbm_metrics.json` — independent test metrics per fold.
- `*_configuration.json` — command-line inputs and PyTorch version.

Metrics include accuracy, balanced accuracy, macro-F1, quadratic weighted kappa, all five class precision/recall/F1/support values, a 5×5 confusion matrix, and the count of **severity underestimations**. Validation/test output also includes the top-class **10-bin expected calibration error (ECE)** and mean maximum softmax score as descriptive uncertainty/reliability indicators. ECE does not make predictions calibrated. The **test JSON** additionally includes per-image IDs, actual/predicted grades, top-class scores and (for CBM) concept scores, allowing identification of representative errors and severe underestimations. These per-image results are local and excluded from Git by default. No metrics are supplied or fabricated in this repository; regenerate them from your authorised dataset and actual checkpoints.

## 5. Generate explanations and example outputs

```bash
python explain.py --checkpoint outputs/cbm_resnet152/fold_1/best_cbm_model.pt --image "D:/Datasets/MMRDR-CFP/img/ts000001.jpg" --output-dir outputs/explanations --concept all
```

The command saves:

- `ts000001_prediction.json` with the predicted grade, five uncalibrated softmax scores and seven image-level concept scores;
- `ts000001_concept_MA.png` and six further side-by-side **original / concept evidence** images.

Optional controls: `--opacity 0.65`, `--threshold 0.25`, `--concept 0` (only microaneurysms), `--device cpu`, or `--max-display-size 1200`. These visualisations show learned spatial activation, not pixel-level ground truth, causal attribution, or verified lesion locations. Use multiple representative correct predictions, errors and severity underestimations for reporting; share illustrative CFP images only if dataset terms permit it.

## 6. Reproducibility and interpretation notes

1. Keep the original `FP.csv` and `img/` together; preserve the dataset's `tr`/`ts` partition.
2. Use the **same** `--seed`, `--folds`, image preprocessing, concept order and output root for both models. The code seeds Python, NumPy and PyTorch; add `--deterministic` for stricter algorithms where supported. Absolute bitwise reproducibility may still vary by hardware and library versions.
3. Training uses grade-stratified five-fold CV on the `tr` subset, with training-only image augmentation and early stopping on validation loss.
4. Choose hyperparameters using training/validation information only. Test images are not used for fitting or checkpoint selection.
5. Record the actual CLI command, software versions, checkpoint provenance, selected folds, and the output JSON files with the technical report.
6. Do not interpret a high softmax score as a calibrated clinical probability; inspect failure cases, including underestimation of severe DR.

The two-stage CBM employs **seven binary concepts**: microaneurysms; hard exudates; intraretinal haemorrhages; venous beading/IRMA; neovascularisation; vitreous haemorrhage; retinal detachment. The five output grades are No DR, Mild NPDR, Moderate NPDR, Severe NPDR and Proliferative DR.

## 7. Data privacy, provenance and limitations

- **Never commit** dataset files, images, `FP.csv`, checkpoints, private paths, personal metadata, results derived from restricted images, or local environments. `.gitignore` excludes likely sensitive directories, but inspect `git status` before each push.
- Pretrained backbone uses torchvision ImageNet weights if `--pretrained` (default for black-box training); first use may require internet. Cite PyTorch/torchvision and the original ResNet literature as appropriate.
- No external clinical validation is presented. The repository is a research prototype, not a clinical screening or diagnostic tool.
- No public licence is declared pending the project owner's/PI's decision. Do not assume permission to reuse or redistribute third-party data or checkpoints.

### References

- Tang, Z. et al. (2026). *A multimodal retinal image dataset for diabetic retinopathy detection using foundation models*. Scientific Data 13, 639. https://doi.org/10.1038/s41597-026-07005-9
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Koh, P. W. et al. (2020). Concept Bottleneck Models. ICML.


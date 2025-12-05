# Stationary Human Presence Detection using 24 GHz FMCW Radar  

This repository contains the full Python software stack that I developed for my Dalyell research project on stationary human presence detection using a commercial 24 GHz frequency-modulated continuous-wave (FMCW) radar.  
The central goal of this project was to evaluate two distinct detection paradigms:

1. **Classical threshold-based signal processing**  
   using spectral features derived from temporal phase variations in the radar return.

2. **Unsupervised machine-learning (ML) detection**  
   using a PCA + Isolation Forest anomaly-detection pipeline trained only on non-human examples.

To support this investigation, I developed a fully custom Python radar interface, capable of:

- Configuring the SiRad Easy r4 radar over UART  
- Receiving and streaming I/Q ADC data in real time  
- Performing FFTs, range gating, and weighted complex summation  
- Extracting the unwrapped temporal phase ϕ(t)  
- Saving timestamped CSV data for further analysis  

The detection pipelines in this repository reproduce exactly the analyses reported in the final research article, including the feature-extraction, Youden’s-J threshold optimisation, model training, and evaluation.

---

# Repository Structure

```text
scripts/
│
├── ADC_FFTs_07OCT.py
├── ADC_FFTs_30SEP_rangeheader.py
├── development_thresholds.py
├── train_iforest.py
└── apply_iforest.py

data/
    raw data will be uploaded at a later date

models/
    iforest_PCA_negonly.joblib

report/
   Yunki Yau Dalyell Report 12NOV25.pdf
```

Each script is documented below.

---

#  1. `ADC_FFTs_07OCT.py` — Real-Time Radar Interface (Main Acquisition Script)

**Purpose:**  
Primary data-acquisition program used in the report.

This script:

- Connects to the SiRad Easy r4 radar via UART  
- Sends configuration commands  
- Streams I/Q ADC samples in real time  
- Computes:
  - FFT(I), FFT(Q)  
  - Complex spectrum |FFT(I + jQ)|  
  - Weighted complex sum Z(t) inside a 1.3–1.7 m range gate  
- Performs phase unwrapping and displays:
  - Live ADC plot  
  - Live FFTs  
  - Live complex spectrum  
  - Live unwrapped ϕ(t)

**Recording to CSV includes:**

- Time (s)  
- Wrapped and unwrapped phase  
- |Z| magnitude  
- 257 positive-frequency FFT bins with **range annotations**

**Keyboard shortcuts:**

- **R** = Start/stop recording  
- **D** = Toggle DC removal  
- **W** = Toggle complex windowing  
- **Q** = Quit  

This is the main script used to generate all raw data for the project.

---

# 2. `development_thresholds.py` — Threshold Optimisation (Youden’s J)

**Purpose:**  
Implements the classical threshold-based classifier used in the paper.

What it does:

- Loads positive and negative feature CSVs  
- Extracts hand-engineered features:  
  - Spectral flatness  
  - Crest factor  
  - Spectral centroid (Hz)  
  - Phase variance (area under Welch PSD)  
- Tests all candidate thresholds between unique values  
- Checks both threshold directions (`>=` and `<=`)  
- Selects the rule that maximises **Youden’s J = TPR – FPR**  
- Computes:
  - Confusion matrix  
  - Sensitivity, specificity  
  - Accuracy  
  - J statistic  
- Saves:
  - Summary CSV of thresholds  
  - Histogram plots for each feature

This script produced the threshold values quoted in the report.

---

# 3. `train_iforest.py` — Train PCA + Isolation Forest Model (Unsupervised ML)

**Purpose:**  
Train the anomaly-detection model described in Section II-D.2 of the report.

Pipeline:

1. Read development CSVs  
2. Extract FFT-bin vectors
3. Per-record L2 normalisation  
4. StandardScaler  
5. Optional PCA (e.g., 95% variance retained)  
6. Train Isolation Forest  
   - Fully unsupervised 
   - The model used in the report was trained on negative (non-human) samples only

Outputs:

- `.joblib` model containing:
  - `"scaler"`
  - `"pca"` (or `None`)
  - `"iforest"`
  - `"feature_from_col"`  
- `dev_predictions_if.csv`  
- `dev_summary_if.html`  
- Optional PCA-2D projection figure

---

#  4. `apply_iforest.py` — Apply Trained Model to Evaluation Dataset

**Purpose:**  
Runs the trained Isolation Forest model on unseen evaluation data.

This script:

- Loads the `.joblib` model  
- Extracts averaged FFT-bin vectors  
- Standardises → PCA transforms → Isolation Forest predicts  
- Converts IF prediction:
  - `+1` → **0 = Non-human**
  - `–1` → **1 = Human present**
- Infers ground-truth labels automatically by:
  - `eval_pos/` and `eval_neg/` directory membership  
- Outputs:
  - `eval_predictions_if.csv`  
  - `eval_summary_if.html` (metrics & confusion matrix)

Matches the evaluation stage from the report.

---

#  Installation

Install all required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn joblib pyserial


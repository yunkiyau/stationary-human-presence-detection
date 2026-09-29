# Stationary Human Presence Detection using 24 GHz FMCW Radar  

This repository contains an end-to-end Python software stack for stationary human presence detection using a commercial 24 GHz frequency-modulated continuous-wave (FMCW) radar. 
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

The detection pipelines in this repository reproduce the analyses reported in the final research article, covering Youden’s-J threshold optimisation, model training, and evaluation. The threshold script consumes precomputed feature tables; the full feature-extraction workflow and complete recording collection are not included.

---

## Repository Structure

```text
src/
│
├── radar_acquisition_fft.py
├── threshold_optimisation.py
├── train_iforest.py
└── apply_iforest.py

data/
    sample_recording.csv

models/
    iforest_PCA_negonly.joblib

report/
   Stationary_Human_Presence_Detection_using_24_GHz_radar21NOV25.pdf
```

## Report and available data

The [final project report](report/Stationary_Human_Presence_Detection_using_24_GHz_radar21NOV25.pdf) describes 60 recordings: 44 for development and 16 for evaluation. Only one example recording is included here, so this checkout alone cannot reproduce the reported performance metrics. Supply your own development/evaluation folders and precomputed positive/negative feature CSVs for the analysis commands below. These are real hardware measurements; no synthetic data is presented as research evidence.

## Installation

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The radar communicates with the host machine over a UART serial interface.
Ensure that the appropriate serial drivers are installed and that the radar is connected and powered on.

## Usage / How to Run

The overall workflow is:

1. Acquire raw radar data and compute FFT/range-gated phase time series
2. Extract features and perform classical threshold-based detection or
3. Train and apply an unsupervised Isolation Forest detector

**Note:** Some scripts require configuration of file paths, serial ports, or analysis parameters directly within the script.

### 1. Real-time acquisition and FFT/range processing

Run the main acquisition script:
```bash
python src/radar_acquisition_fft.py
```

This script streams raw I/Q ADC data from the radar in real time, performs FFT-based range processing, computes the weighted complex sum within a specified range gate, and displays live diagnostic plots.
CSV files containing phase and FFT-bin data can be recorded using the keyboard controls.

Ensure that the correct serial port and radar configuration parameters are set in the script before running.

### 2. Classical threshold-based detection

Run the threshold optimisation and evaluation pipeline:
```bash
python src/threshold_optimisation.py --neg dev_neg_features.csv --pos dev_pos_features.csv
```

This script loads extracted feature CSVs, evaluates candidate thresholds for each feature, and selects the rule that maximises Youden’s J statistic.
It generates summary CSVs, histograms, and performance metrics consistent with the reported results.

### 3. Unsupervised ML detection (Isolation Forest)

Train the model
```bash
python src/train_iforest.py --dev_dir dev_raw --dev_posdir dev_pos --dev_negdir dev_neg --pca_var 0.95 --out_model iforest_model.joblib
```

This script trains a PCA + Isolation Forest anomaly detector using development data (negative samples only, as reported).
The trained model is saved as a .joblib file for later use.

Apply the trained model
```bash
python src/apply_iforest.py --model iforest_model.joblib --eval_dir eval_raw --eval_posdir eval_pos --eval_negdir eval_neg
```

This script applies the trained model to unseen evaluation data and outputs prediction CSVs and HTML summaries containing performance metrics and confusion matrices.

## Script Documentation

Each script is documented below.

---

### `radar_acquisition_fft.py` — Real-Time Radar Interface (Main Acquisition Script)

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

### `threshold_optimisation.py` — Threshold Optimisation (Youden’s J)

**Purpose:**  
Implements the classical threshold-based classifier used in the paper.

What it does:

- Loads positive and negative feature CSVs  
- Uses precomputed hand-engineered feature columns:  
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

### `train_iforest.py` — Train PCA + Isolation Forest Model (Unsupervised ML)

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

###  `apply_iforest.py` — Apply Trained Model to Evaluation Dataset

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

## Acknowledgements
This work was conducted under supervision at the Australian Centre for Robotics.
Supervised by Dr. Graham Brooker.

Code and documentation are shared for academic review and demonstration purposes.
Please contact the author before reusing or redistributing any portion of this work.



## Offline regression checks

After installing the requirements, run:

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic data or simulated instruments. They verify software behaviour, not hardware operation or the original experimental results.

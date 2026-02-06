# AirTremor Artifact

This repository contains the code used in our paper to extract **power modulation** residuals from Wi-Fi CSI and to evaluate the decoding model. And it will show SER(y_acc). The training model and data used for testing are from one of the test runs on the Test Device described in Section 6.3.2. If wanna train model, prepare data structured according to dataset.py as specified in utils.py, then execute train.py. 

## Overview

The pipeline consists of two stages:

1. **Power Modulation Residual Extraction**
   This stage processes raw CSI measurements and suppresses dominant channel components to extract residual features correlated with power modulation.

2. **Model Evaluation**
   This stage evaluates the trained model on the extracted residuals and reports decoding performance.

## Usage

### Step 1: Residual Extraction

Run the following command to process raw CSI data and extract power modulation residuals:

```bash
python main.py
```
### Step 2: Model Evaluation

After residual extraction, run:

```bash
python eval.py
```

`eval.py` loads the extracted features and evaluates the trained model, reporting y_acc(SER).



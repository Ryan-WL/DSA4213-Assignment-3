# DSA4213-Assignment-3

# BioBERT Text Classification on PubMed 20k RCT Dataset

This repository contains code to fine-tune the **BioBERT** pretrained transformer model on the **PubMed 20k RCT** dataset for **biomedical text classification**.

---

## Project Overview

- **Model:** BioBERT (dmis-lab/biobert-base-cased-v1.1)
- **Dataset:** PubMed 20k Randomized Controlled Trials (RCT) sentences
- **Task:** Text classification of sentences by section type (Background, Objective, Methods, Results, Conclusions)
- **Training:** Full fine-tuning on GPU and Low-Rank Adpatation (LoRA)
- **Evaluation metrics:** Accuracy, Macro F1 score, Confusion Matrix

---

## Repository Structure
``` shell
DSA4213-Assignment-3/
├── main.py
├── preparation.py
├── README.md
├── requirements.txt
├── test.py
├── train.py
└── train_lora.py
```

---

## Setup

### 1. Clone the repository

```bash
cd Desktop (put in the whole path file)
git clone https://github.com/Ryan-WL/DSA4213-Assignment-3
cd (path to cloned repo)\DSA4213-Assignment-3
```
### 2. Install dependencies
Open command prompt as administrator, and install necessary dependencies by doing the following:
```bash
pip install -r requirements.txt
```
### 3. Dataset
1. Open the file preparation.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. The script will automatically download the dataset into the data/ folder, with data already split into train, val and test for future uses.

## Fine-tuning and Evaluation
1. Open the file main.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. Script will fune-tune the models and will print the accuracy, F1 scores and show the Confusion Matrix accordingly

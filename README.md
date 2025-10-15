# DSA4213-Assignment-3

# BioBERT Text Classification on PubMed 20k RCT Dataset

This repository contains code to fine-tune the **BioBERT** pretrained transformer model on the **PubMed 20k RCT** dataset for **biomedical text classification**.

---

## Project Overview

- **Model:** BioBERT (dmis-lab/biobert-base-cased-v1.1)
- **Dataset:** PubMed 20k Randomized Controlled Trials (RCT) sentences
- **Task:** Text classification of sentences by section type (e.g., Background, Methods, Results, Conclusions)
- **Training:** Full fine-tuning on GPU (if available)
- **Evaluation metrics:** Accuracy, Macro F1 score

---

## Repository Structure
``` shell
DSA4213-Assignment-3/
├── main.py
├── preparation.py
├── requirements.txt
├── README.md
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
## Fine-tuning
1. Open the file main.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. F1 score of BioBERT on test dataset will be printed, along with the confusion matrix

## Dataset
1. Open the file src/preparation.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. The script will automatically download the dataset into the data/ folder, with data already split into train, val and test for future uses.

## Training of Model
1. Open the file src/train.py in Python IDLE.
2. Press F5 or go to Run > Run Module.

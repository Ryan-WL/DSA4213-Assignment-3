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
your-repo/
├── data/
├── scripts/
│ └── download_data.py
├── src/
│ ├── data_loader.py
│ ├── train.py
│ └── evaluate.py
├── notebooks/
├── requirements.txt
├── README.md
└── main.py
```

---

## Setup

### 1. Clone the repository

```bash
cd Desktop
git clone https://github.com/Ryan-WL/DSA4213-Assignment-3
```
### 2. Install dependencies
Open command prompt as administrator, and install necessary dependencies by doing the following:
```bash
pip install -r requirements.txt
```

## Dataset
1. Open the file src/download_dataset.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. The script will automatically download the dataset into the data/ folder.

## Training of Model
1. Open the file src/train.py in Python IDLE.
2. Press F5 or go to Run > Run Module.

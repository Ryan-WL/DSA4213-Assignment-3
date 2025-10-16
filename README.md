# DSA4213-Assignment-3

# BioBERT and BERT Text Classification on PubMed 20k RCT Dataset

This repository contains code to fine-tune the **BioBERT** and **base BERT** pretrained transformer model on the **PubMed 20k RCT** dataset for **biomedical text classification**.

---

## Project Overview

- **Model:** BioBERT (dmis-lab/biobert-base-cased-v1.1) and base BERT ("bert-base-uncased")
- **Dataset:** PubMed 20k Randomized Controlled Trials (RCT) sentences
- **Task:** Text classification of sentences by section type (Background, Objective, Methods, Results, Conclusions)
- **Training:** Full fine-tuning on GPU and Low-Rank Adpatation (LoRA) also on GPU
- **Evaluation metrics:** Accuracy, Macro F1 score, Confusion Matrix

---

## Repository Structure
``` shell
DSA4213-Assignment-3/
├── README.md
├── main.py
├── preparation.py
├── requirements.txt
├── test.py
├── train.py
└── train_lora.py
```

---

## Setup

### 1. Clone the repository

```bash
cd Desktop (whole path file)
git clone https://github.com/Ryan-WL/DSA4213-Assignment-3
```
### 2. Install dependencies
Open command prompt as administrator, and install necessary dependencies by doing the following:
```bash
pip install -r requirements.txt
```
### 3. Dataset
1. Open the file preparation.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. The script will automatically download the dataset into the data/ folder, with data already split into train, val and test for future uses. Below is what files and folders should appear after running the preparation.py script.

``` shell
DSA4213-Assignment-3/
├── data
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── README.md
├── label_encoder.pkl
├── main.py
├── num_labels.txt
├── preparation.py
├── requirements.txt
├── test.py
├── test_data.pkl
├── train.py
├── train_data.pkl
├── train_lora.py
└── val_data.pkl
```

## Fine-tuning and Evaluation
1. Open the file main.py in Python IDLE.
2. Press F5 or go to Run > Run Module.
3. Script will fune-tune the models and will print the accuracy, F1 scores and show the Confusion Matrix accordingly. Warnings will pop up but do not worry, it is just a reminder to train our own weights. To continue the code after plotting confusion matrix, close the current confusion matrix to allow the code to continue running. All the previous confusion matrices are saved locally in the same file labelled as "confusion_matrix_{transformer} with {finetune}" if past confusion matrices are required.

## 📊 Results

| Model       | Fine-tuning | Accuracy | F1 (Macro) |
|-------------|-------------|----------|------------|
| BioBERT     | Full-tune   |  0.85    | 0.79       |
| BioBERT     | LoRA        |  0.83    | 0.78       |
| BERT        | Full-tune   |  0.81    | 0.75       |
| BERT        | LoRA        |  0.75    | 0.65       |

Confusion matrices are saved in as individual png images.

## Acknowledgements

- Dataset: [PubMed 20k RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) by Franck Dernoncourt et al.
- Source: Based on data from the [PubMed Baseline Dataset (2016)](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/), provided by the U.S. National Library of Medicine (NLM).

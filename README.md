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
.
├── data/                   # Dataset files (gitignored, downloaded via script)
├── src/                    # Source code
│   ├── data_loader.py      # Dataset download & loading functions
│   ├── train.py            # Training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file


---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>

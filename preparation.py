import os
import requests
import torch
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# ==== Constants ====
BASE_URL = "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR = "data"

FILES = {
    "train": "train.txt",
    "dev": "dev.txt",   # Will be saved locally as val.txt
    "test": "test.txt",
}

# ==== Downloading Functionality ====
def download_file(url: str, save_path: str):
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved to {save_path}")

def download_dataset():
    """
    Download dataset files and save them in the 'data/' folder.
    Renames dev.txt to val.txt.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    for split_name, filename in FILES.items():
        url = BASE_URL + filename
        save_name = "val.txt" if split_name == "dev" else filename
        save_path = os.path.join(DATA_DIR, save_name)

        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping download.")
        else:
            download_file(url, save_path)

# ==== Data Loading Functionality ====
def load_split(split_name):
    """
    Load a split file (train/val/test) from data folder and return list of dicts with 'sentence' and 'label'.
    """
    file_map = {
        "train": "train.txt",
        "val": "val.txt",    # renamed version of dev.txt
        "test": "test.txt",
    }
    
    max_experiments = 500 if split_name == "train" else 65
    filepath = os.path.join(DATA_DIR, file_map[split_name])
    data = []
    experiment_count = 0
    reading = False  # only start reading data after '###'

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("###"):
                experiment_count += 1
                if experiment_count > max_experiments:
                    break
                reading = True
                continue

            if not line or not reading:
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                label, sentence = parts
            else:
                label = parts[0]
                sentence = parts[-1]

            data.append({"sentence": sentence, "label": label})

    return data

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.labels = self.label_encoder.transform([item["label"] for item in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# ==== Entry Point ====
if __name__ == "__main__":
    download_dataset()
    
    # Load all splits
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")

    with open("train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open("val_data.pkl", "wb") as f:
        pickle.dump(val_data, f)

    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)


    # Prepare label encoder based on train labels only
    label_encoder = LabelEncoder()
    label_encoder.fit([item["label"] for item in train_data])
    num_labels = len(label_encoder.classes_)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open("num_labels.txt", "w") as f:
        f.write(str(num_labels))


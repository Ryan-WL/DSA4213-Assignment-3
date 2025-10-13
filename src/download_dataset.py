import os
import requests

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
    filepath = os.path.join(DATA_DIR, file_map[split_name])
    data = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("###"):  # skip empty lines or abstract markers
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                label, sentence = parts
            else:
                label = parts[0]
                sentence = parts[-1]
            data.append({"sentence": sentence, "label": label})
    return data

# ==== Entry Point ====
if __name__ == "__main__":
    download_dataset()

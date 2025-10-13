import os
import requests

BASE_URL = "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
FILES = {
    "train": "train.txt",
    "dev": "dev.txt",
    "test": "test.txt",
}

def download_file(url: str, save_path: str):
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved to {save_path}")

def main():
    os.makedirs("data", exist_ok=True)

    for split_name, filename in FILES.items():
        url = BASE_URL + filename
        # Rename dev.txt to val.txt locally
        if split_name == "dev":
            save_name = "val.txt"
        else:
            save_name = filename

        save_path = os.path.join("data", save_name)
        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping download.")
        else:
            download_file(url, save_path)

if __name__ == "__main__":
    main()

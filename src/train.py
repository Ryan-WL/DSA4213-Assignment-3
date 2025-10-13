import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from src.data_loader import download_dataset, load_split

# =======================
# Device setup
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================
# Parameters
# =======================
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
NUM_LABELS = 5  # Adjust if you have a different number of labels
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# =======================
# Dataset Class
# =======================
class PubMedDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sentence = sample["sentence"]
        label = self.label2id[sample["label"]]

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# =======================
# Load data
# =======================
download_dataset()
train_data = load_split("train")
val_data = load_split("val")

# Get all unique labels
all_labels = sorted(list(set([ex["label"] for ex in train_data])))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}

# =======================
# Tokenizer and datasets
# =======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = PubMedDataset(train_data, tokenizer, label2id)
val_dataset = PubMedDataset(val_data, tokenizer, label2id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =======================
# Model setup
# =======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# =======================
# Training loop
# =======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

    # =======================
    # Evaluation
    # =======================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

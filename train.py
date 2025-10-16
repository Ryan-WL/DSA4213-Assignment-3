import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from preparation import load_split, TextClassificationDataset
from peft import get_peft_model, LoraConfig, TaskType

def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def train(transformer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load num_labels
    with open("num_labels.txt", "r") as f:
        num_labels = int(f.read().strip())

    # Load tokenizer and model from BioBERT
    if transformer=="BioBERT":
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        save_path="best_model_biobert.pt"
    elif transformer=="BERT":
        model_name = "bert-base-uncased"
        save_path="best_model_base_bert.pt"
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Load  label_encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load train_data
    with open("train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    # Load val_data
    with open("val_data.pkl", "rb") as f:
        val_data = pickle.load(f)

    # Create Dataset and DataLoader
    train_dataset = TextClassificationDataset(train_data, tokenizer, label_encoder)
    val_dataset = TextClassificationDataset(val_data, tokenizer, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    f1_score=0

    print(f"{transformer} with full-tuning training:")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = logits.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc, f1 = compute_metrics(all_preds, all_labels)

        print("Validation done")
        print(f1)

        if f1>f1_score:
            f1_score=f1
            torch.save(model.state_dict(), save_path)
        else:
            break
        
    print(f"Best {transformer} model saved to {save_path}")


#if __name__ == "__main__":
#    train()

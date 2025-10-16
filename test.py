import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preparation import TextClassificationDataset, load_split
from train import compute_metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from peft import PeftModel

def test(transformer, finetune):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load num_labels
    with open("num_labels.txt", "r") as f:
        num_labels = int(f.read().strip())

    if transformer=="BioBERT":
        model_name = "dmis-lab/biobert-base-cased-v1.1"
    elif transformer=="BERT":
        model_name = "bert-base-uncased"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load saved weights and model depending on transformer and finetune
    if finetune == "full-tune":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if transformer == "BioBERT":
            best_weights = "best_model_biobert.pt"
        elif transformer == "BERT":
            best_weights = "best_model_base_bert.pt"

        model.load_state_dict(torch.load(best_weights, map_location=device))
            
    elif finetune == "LoRA":
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if transformer == "BioBERT":
            best_weights = "best_model_lora_biobert"
        elif transformer == "BERT":
            best_weights = "best_model_lora_base_bert"
        
        model = PeftModel.from_pretrained(base_model, best_weights)

    # Load the saved weights (state_dict)
    model.to(device)
    model.eval()

    # Load  label_encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load val_data
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    # Prepare test dataset and loader
    test_dataset = TextClassificationDataset(test_data, tokenizer, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=32)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
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
    print(f"Evaluation of {transformer} with {finetune} strategy:")
    print(f"Test Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")

    # Show confusion matrix
    predicted_labels = np.argmax(all_preds, axis=1)
    cm = confusion_matrix(all_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix of {transformer} with {finetune}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{transformer} with {finetune}.png", dpi=300, bbox_inches='tight')
    plt.show()

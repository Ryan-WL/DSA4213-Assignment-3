from train import train
from train_lora import train_lora
from test import test

def main():
    # BioBERT finetune with full-finetuning and LoRA
    train("BioBERT")

    train_lora("BioBERT")

    # Evaluation of both finetuning strategies with test dataset
    test("BioBERT", "full-tune")
    test("BioBERT", "LoRA")
    
    # Base BERT finetune with full-finetuning and LoRA
    train("BERT")

    train_lora("BERT")

    # Evaluation of both finetuning strategies with test dataset
    test("BERT", "full-tune")
    test("BERT", "LoRA")

    return 0

if __name__ == "__main__":
    main()

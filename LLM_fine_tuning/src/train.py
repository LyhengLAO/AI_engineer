"""
train.py
--------
Fine-tuning LoRA/QLoRA d'un petit LLM (par défaut Qwen2.5-1.5B-Instruct)
sur la tâche Text-to-SQL (dataset b-mc2/sql-create-context).

⚠️ RESSOURCE: nécessite un GPU (T4 minimum). Conçu pour Google Colab.
   Ne PAS lancer sur un PC local AMD (bitsandbytes ne supporte pas ROCm
   de façon fiable).

Usage (dans Colab, après upload de data/processed/*.jsonl):
    !python src/train.py \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --train_file data/processed/train.jsonl \
        --val_file data/processed/val.jsonl \
        --output_dir models/lora_adapters \
        --epochs 2
"""

import argparse

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import torch

from prompts import build_training_text


def format_example(example):
    """Formate en template chat (format partagé avec l'inférence)."""
    return {"text": build_training_text(example["instruction"], example["output"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/lora_adapters")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    print(f"Chargement du modèle {args.model_name} en 4-bit (QLoRA)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Chargement des datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    dataset = dataset.map(format_example)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )

    print("Début de l'entraînement...")
    trainer.train()

    print(f"Sauvegarde des adapters LoRA dans {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Terminé. Télécharge le dossier depuis Colab avant de fermer la session:")
    print(f"  from google.colab import files")
    print(f"  !zip -r lora_adapters.zip {args.output_dir}")
    print(f"  files.download('lora_adapters.zip')")


if __name__ == "__main__":
    main()

# models/train_fanar_lora.py
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from zipfile import ZipFile


def main():
    # ========== Load Data ==========
    train_df = pd.read_csv("/kaggle/input/aiiidata/ground_truth.csv")
    dev_df = pd.read_csv("/kaggle/input/testset/test_unlabeled.csv")

    # Format for instruction tuning
    train_data = train_df.dropna(subset=["content", "Class"])
    train_data = train_data.apply(lambda row: {
        "text": f"<|user|>\n{row['content']}\n<|assistant|>\n{row['Class']}"
    }, axis=1).tolist()
    train_dataset = Dataset.from_list(train_data)

    # ========== Load Tokenizer ==========
    model_name = "QCRI/Fanar-1-9B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== Quantization Config ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # ========== Load Model ==========
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # ========== Tokenization ==========
    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ========== Training Arguments ==========
    training_args = TrainingArguments(
        output_dir="./fanar-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Train
    trainer.train()

    # ========== Inference Function ==========
    def generate_prediction(text):
        prompt = f"<|user|>\n{text}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return "machine" if "machine" in decoded.lower() else "human"

    # Predict on dev set
    dev_df["label"] = dev_df["content"].apply(generate_prediction)

    # Save predictions
    dev_df[["id", "label"]].to_csv("predictions.csv", index=False)

    # Zip predictions
    with ZipFile("predictions.zip", "w") as zipf:
        zipf.write("predictions.csv")


if __name__ == "__main__":
    main()

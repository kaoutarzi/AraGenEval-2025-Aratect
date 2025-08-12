
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile


def main():
    # ========== Load Data ==========
    train_df = pd.read_csv("/kaggle/input/aidata/ground_truth.csv")
    testset_df = pd.read_csv("/kaggle/input/aidata/test_unlabeled.csv")

    # Drop missing values in important columns
    train_df = train_df.dropna(subset=["content", "Class"])

    # Encode labels (human=0, machine=1)
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["Class"])

    # Create HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[["content", "label"]])

    # ========== Load Tokenizer ==========
    model_name = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=512)

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # ========== Load Model ==========
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ========== Training Arguments ==========
    training_args = TrainingArguments(
        output_dir="./arabert-output",
        per_device_train_batch_size=4,
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
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # ========== Inference Function ==========
    def classify(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            return label_encoder.inverse_transform([predicted_class_id])[0]

    # Apply classification to test set
    testset_df["label"] = testset_df["content"].apply(classify)

    # Save predictions
    testset_df[["id", "label"]].to_csv("predictions.csv", index=False)

    # Zip predictions
    with ZipFile("predictions.zip", "w") as zipf:
        zipf.write("predictions.csv")


if __name__ == "__main__":
    main()

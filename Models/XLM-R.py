# models/train_xlm_roberta.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import zipfile
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # ========== Load Data ==========
    train_df = pd.read_csv("/kaggle/input/aidata/ground_truth.csv")
    train_df = train_df.dropna(subset=["content", "Class"])
    train_df["label"] = train_df["Class"].map({"human": 0, "machine": 1})

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Tokenize
    encodings = tokenizer(
        train_df["content"].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    labels = torch.tensor(train_df["label"].tolist())

    train_dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # ========== Load Model ==========
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # ========== Training Loop ==========
    epochs = 3
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for input_ids, attn_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {train_acc:.4f}")

    # ========== Predict on Dev Set ==========
    testset_df = pd.read_csv("/kaggle/input/aidata/test_unlabeled.csv")
    testset_texts = dev_df["content"].tolist()

    testset_enc = tokenizer(dev_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    testset_dataset = TensorDataset(testset_enc["input_ids"], testset_enc["attention_mask"])
    testset_loader = DataLoader(testset_dataset, batch_size=4)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for input_ids, attn_mask in dev_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    testset_df["label"] = [ "human" if p == 0 else "machine" for p in all_preds ]
    testset_df = dev_df.sort_values("id").reset_index(drop=True)

    # Save predictions
    testset_df[["id", "label"]].to_csv("predictions.csv", index=False)

    # Zip
    with zipfile.ZipFile("pred.zip", "w") as zipf:
        zipf.write("predictions.csv")


if __name__ == "__main__":
    main()

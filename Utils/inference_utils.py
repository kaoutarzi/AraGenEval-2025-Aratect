
import torch
import pandas as pd
from zipfile import ZipFile


def classify_texts(model, tokenizer, texts, label_encoder, max_length=512):
    """
    Classify a list of texts using a trained HuggingFace classification model.
    Returns predicted labels (decoded from label_encoder).
    """
    preds = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        preds.append(label_encoder.inverse_transform([pred_id])[0])
    return preds


def generate_predictions_causal(model, tokenizer, texts, max_new_tokens=10):
    """
    Generate predictions using a causal LM (e.g., Fanar model).
    Returns 'human' or 'machine'.
    """
    preds = []
    for text in texts:
        prompt = f"<|user|>\n{text}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append("machine" if "machine" in decoded.lower() else "human")
    return preds


def save_predictions(ids, labels, filename_csv="predictions.csv", filename_zip="predictions.zip"):
    """
    Save predictions to CSV and ZIP for submission.
    """
    df = pd.DataFrame({"id": ids, "label": labels})
    df.to_csv(filename_csv, index=False)
    with ZipFile(filename_zip, "w") as zipf:
        zipf.write(filename_csv)

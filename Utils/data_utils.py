
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer


def load_and_encode_classification_data(train_path, text_col, label_col, model_name):
    """
    Load CSV, drop missing rows, encode labels, return HuggingFace Dataset and label encoder.
    """
    df = pd.read_csv(train_path).dropna(subset=[text_col, label_col])
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[label_col])

    dataset = Dataset.from_pandas(df[[text_col, "label"]]])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch[text_col], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return dataset, label_encoder, tokenizer


def load_for_pytorch(train_path, text_col, label_col, model_name, label_map=None):
    """
    Load CSV, drop missing rows, map labels manually if given, return encodings & labels.
    """
    df = pd.read_csv(train_path).dropna(subset=[text_col, label_col])

    if label_map:
        df["label"] = df[label_col].map(label_map)
    else:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df[label_col])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(df[text_col].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

    return encodings, df["label"].tolist()

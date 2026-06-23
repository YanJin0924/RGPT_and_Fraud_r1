import torch
import numpy as np
import joblib
import pandas as pd
import os

os.makedirs("mid_result", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_models = joblib.load("Rgpt_BaseModels/base_models.pkl")
model_alphas = joblib.load("Rgpt_BaseModels/model_alphas.pkl")
tokenizer = joblib.load("Rgpt_BaseModels/tokenizer.pkl")
MAX_LEN = 64

def rgpt_predict(text_list):
    processed_texts = []
    for text in text_list:
        if isinstance(text, list):
            processed_texts.append(" ".join(text))
        else:
            processed_texts.append(str(text))

    total_logits = 0
    for model, alpha in zip(base_models, model_alphas):
        model.eval()
        inputs = tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        total_logits += logits * alpha

    return np.argmax(total_logits, axis=1)


def test_dataset(name, path):
    X, y_true = joblib.load(path)
    y_pred = rgpt_predict(X)
    # 保存结果
    pd.DataFrame({
        "true_label": y_true,
        "pre_label": y_pred
    }).to_csv(f"mid_result/{name}.csv", index=False)

test_list = [
    ("original", "train_dataset/test.pkl"),
    ("aug_信任建立", "train_dataset/aug_信任建立.pkl"),
    ("aug_紧迫感", "train_dataset/aug_紧迫感.pkl"),
    ("aug_情感操纵", "train_dataset/aug_情感操纵.pkl"),
    ("aug_权威伪装", "train_dataset/aug_权威伪装.pkl"),
    ("aug_利益诱惑", "train_dataset/aug_利益诱惑.pkl"),
    ("aug_模糊性诱导", "train_dataset/aug_模糊性诱导.pkl")
]

for name, path in test_list:
    test_dataset(name, path)
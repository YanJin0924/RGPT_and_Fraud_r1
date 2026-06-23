import torch
import numpy as np
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# 配置
os.makedirs("Rgpt_BaseModels", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
EPOCHS = 1
BATCH_SIZE = 16
K = 2
MAX_LEN = 64


class TextDataset(Dataset):
    def __init__(self, texts, labels, weights=None):
        self.texts = texts
        self.labels = [int(label) for label in labels]
        # 🔥 核心修复1：初始权重=1，不除以样本数（防止数值过小）
        self.weights = weights if weights is not None else np.ones(len(texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.weights[idx]

# 批处理
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    weights = [item[2] for item in batch]
    inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return inputs, torch.tensor(labels, dtype=torch.long), torch.tensor(weights)

# 加载数据
X_train, y_train = joblib.load("train_dataset/train.pkl")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# RGPT 核心算法
n_samples = len(X_train)
sample_weights = np.ones(n_samples)
base_models = []
model_alphas = []

for k in range(K):
    print(f"\nRGPT 训练第 {k + 1} 个基学习器")
    train_set = TextDataset(X_train, y_train, sample_weights)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels, w in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            w = w.to(device)

            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            loss = (loss_fn(outputs.logits, labels) * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    # 预测
    model.eval()
    all_pred = []
    with torch.no_grad():
        for inputs, _, _ in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred = torch.argmax(model(**inputs).logits, dim=1).cpu().numpy()
            all_pred.extend(pred)

    y_np = np.array([int(label) for label in y_train])
    err_rate = np.sum(all_pred != y_np) / n_samples

    alpha = np.log((1 - err_rate) / (err_rate + 1e-8)) if err_rate < 0.5 else 0.0
    base_models.append(model)
    model_alphas.append(alpha)
    print(f"错误率: {err_rate:.4f} | 模型权重: {alpha:.4f}")


    for i in range(n_samples):
        sample_weights[i] *= np.exp(alpha if all_pred[i] != y_np[i] else -alpha)

    sample_weights = sample_weights / np.sum(sample_weights)

joblib.dump(base_models, "Rgpt_BaseModels/base_models.pkl")
joblib.dump(model_alphas, "Rgpt_BaseModels/model_alphas.pkl")
joblib.dump(tokenizer, "Rgpt_BaseModels/tokenizer.pkl")

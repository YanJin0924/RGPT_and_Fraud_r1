
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import joblib
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
TRAIN_CSV = "训练集结果.csv"
TEST_CSV = "测试集结果.csv"
AUGMENTED_DIR = "dataset/augmented"
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["specific_dialogue_content", "is_fraud"])
    df["label"] = df["is_fraud"].astype(int)
    df["clean_content"] = df["specific_dialogue_content"].str.replace("\n", " ").str.replace(r"\s+", " ", regex=True)
    return df[["clean_content", "label"]]

train_df = load_and_clean(TRAIN_CSV)
test_df = load_and_clean(TEST_CSV)

print(f"训练集: {len(train_df)} 样本，正样本比例: {train_df['label'].mean():.2%}")
print(f"测试集: {len(test_df)} 样本，正样本比例: {test_df['label'].mean():.2%}")

class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

train_dataset = FraudDataset(train_df['clean_content'].values, train_df['label'].values, tokenizer)
test_dataset = FraudDataset(test_df['clean_content'].values, test_df['label'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(data_loader)

def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return acc, prec, rec, f1, predictions

print("开始训练BERT模型")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f"Epoch {epoch+1}/{EPOCHS} - 训练损失: {train_loss:.4f}")

# 原始测试集评估
acc, prec, rec, f1, test_preds = evaluate(model, test_loader)
print("\n原始测试集性能（BERT）:")
print(f"准确率: {acc:.4f}")
print(f"精确率: {prec:.4f}")
print(f"召回率: {rec:.4f}")
print(f"F1分数: {f1:.4f}")

test_df['bert_pred'] = test_preds
test_df.to_csv(os.path.join(RESULT_DIR, "bert_original_test_result.csv"), index=False)

model.save_pretrained("model/bert_fraud_model")
tokenizer.save_pretrained("model/bert_fraud_model")
print("模型已保存到 model/bert_fraud_model")

strategy_names = ["信任建立", "紧迫感", "情感操纵", "权威伪装", "利益诱惑", "模糊性诱导"]
all_metrics = [{"测试集类型": "原始测试集", "准确率": acc, "精确率": prec, "召回率": rec, "F1分数": f1}]

print("\nBERT 模型在 Fraud-R1 多轮诱导策略下的鲁棒性评估")

for strategy in strategy_names:
    csv_path = os.path.join(AUGMENTED_DIR, f"{strategy}_clean.csv")
    if not os.path.exists(csv_path):
        print(f"警告: {csv_path} 不存在，跳过该策略。")
        continue
    aug_df = pd.read_csv(csv_path)
    if 'clean_content' not in aug_df.columns:
        print(f"{strategy} 数据缺少 clean_content 列，跳过。")
        continue
    aug_texts = aug_df['clean_content'].values
    if 'label' in aug_df.columns:
        aug_labels = aug_df['label'].values
    else:
        aug_labels = aug_df['is_fraud'].astype(int).values

    aug_dataset = FraudDataset(aug_texts, aug_labels, tokenizer)
    aug_loader = DataLoader(aug_dataset, batch_size=BATCH_SIZE, shuffle=False)

    acc_a, prec_a, rec_a, f1_a, preds_a = evaluate(model, aug_loader)
    print(f"{strategy}策略")
    print(f"准确率: {acc_a:.4f}")
    print(f"精确率: {prec_a:.4f}")
    print(f"召回率: {rec_a:.4f}")
    print(f"F1分数: {f1_a:.4f}")

    all_metrics.append({
        "测试集类型": f"{strategy}策略",
        "准确率": acc_a,
        "精确率": prec_a,
        "召回率": rec_a,
        "F1分数": f1_a
    })

    result_df = pd.DataFrame({"true_label": aug_labels, "bert_pred": preds_a})
    result_df.to_csv(os.path.join(RESULT_DIR, f"bert_{strategy}_result.csv"), index=False)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel(os.path.join(RESULT_DIR, "bert_性能对比表.xlsx"), index=False)

try:
    ada_df = pd.read_excel("result/虚假通话检测模型性能对比表.xlsx")
    combined = pd.merge(ada_df, metrics_df, on="测试集类型", suffixes=("_AdaBoost", "_BERT"))
    combined.to_excel("result/AdaBoost_vs_BERT对比.xlsx", index=False)
    print("已生成AdaBoost vs BERT 对比表：result/AdaBoost_vs_BERT对比.xlsx")
except Exception as e:
    print("未找到 AdaBoost 结果文件，跳过合并对比。")
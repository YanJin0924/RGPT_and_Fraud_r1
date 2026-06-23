import pandas as pd
import joblib
import os
os.makedirs("train_dataset", exist_ok=True)

# 读取原始CSV
train_df = pd.read_csv("训练集结果.csv")
test_df = pd.read_csv("测试集结果.csv")


train_df = train_df.dropna(subset=["specific_dialogue_content", "is_fraud"])
test_df = test_df.dropna(subset=["specific_dialogue_content", "is_fraud"])


X_train = train_df["specific_dialogue_content"].astype(str).tolist()
y_train = train_df["is_fraud"].tolist()
X_test = test_df["specific_dialogue_content"].astype(str).tolist()
y_test = test_df["is_fraud"].tolist()

joblib.dump((X_train, y_train), "train_dataset/train.pkl")
joblib.dump((X_test, y_test), "train_dataset/test.pkl")


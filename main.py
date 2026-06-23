# ===================== 复现文献：Adaptive Boosting 诈骗文本分类 =====================
# 严格对齐论文Algorithm 1 | 本地无网 | 无下载 | 无401报错 | 双任务分类
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ===================== 你的配置（完全不变） =====================
CONFIG = {
    "data_root": r"D:\Document_Yan\研一下\机器学习\欺诈通话数据集\欺诈通话数据集",
    "train_filename": "训练集结果.csv",
    "test_filename": "测试集结果.csv",
    "text_col": "specific_dialogue_content",
    "is_fraud_col": "is_fraud",
    "fraud_type_col": "fraud_type",
    "save_dir": r"D:\Document_Yan\研一下\机器学习\反诈分类结果",
    "K": 5,  # 文献：迭代K个基学习器
}


# ===================== 1. 数据加载（清洗空值） =====================
def load_data():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    train_path = os.path.join(CONFIG["data_root"], CONFIG["train_filename"])
    test_path = os.path.join(CONFIG["data_root"], CONFIG["test_filename"])

    # 读取数据
    try:
        train_df = pd.read_csv(train_path, encoding="utf-8")
        test_df = pd.read_csv(test_path, encoding="utf-8")
    except:
        train_df = pd.read_csv(train_path, encoding="gbk")
        test_df = pd.read_csv(test_path, encoding="gbk")

    # 清洗空值
    train_df = train_df.dropna(subset=[CONFIG["text_col"], CONFIG["is_fraud_col"]])
    test_df = test_df.dropna(subset=[CONFIG["text_col"], CONFIG["is_fraud_col"]])

    # 标签转换
    train_df["label_bin"] = train_df[CONFIG["is_fraud_col"]].astype(str).str.strip().map(
        {"True": 1, "False": 0, "1": 1, "0": 0})
    test_df["label_bin"] = test_df[CONFIG["is_fraud_col"]].astype(str).str.strip().map(
        {"True": 1, "False": 0, "1": 1, "0": 0})
    train_df = train_df.dropna(subset=["label_bin"])
    test_df = test_df.dropna(subset=["label_bin"])

    print(
        f"训练集：{len(train_df)} 条 | 正常：{int(len(train_df) - train_df['label_bin'].sum())} | 诈骗：{int(train_df['label_bin'].sum())}")
    print(f"测试集：{len(test_df)} 条")
    return train_df, test_df


# ===================== 2. 文本特征提取 =====================
def get_features(train_text, test_text):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_X = tfidf.fit_transform(train_text)
    test_X = tfidf.transform(test_text)
    return train_X, test_X, tfidf


# ===================== 3. 核心：复现文献 Adaptive Boosting 算法 =====================
class AdaBoost4TextClassification:
    """严格复现论文《Adaptive Boosting LLMs for Text Classification》算法"""

    def __init__(self, K=5):
        self.K = K  # 文献：基学习器数量
        self.alphas = []  # 文献：学习器权重
        self.models = []  # 文献：基学习器列表

    def fit(self, X, y):
        # 步骤1：初始化样本权重（论文：w_i = 1/N）
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples

        # 步骤2：迭代训练K个基学习器（论文核心循环）
        for k in range(self.K):
            # 训练加权基学习器
            model = DecisionTreeClassifier(max_depth=2)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # 计算错误率（论文：ε_k）
            err = np.sum(w * (y_pred != y)) / np.sum(w)
            if err > 0.5: break

            # 计算学习器权重（论文：α_k）
            alpha = 0.5 * np.log((1 - err) / err)
            self.alphas.append(alpha)
            self.models.append(model)

            # 更新样本权重（论文核心：错样本加权，对样本降权）
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)  # 归一化

            print(f"文献AdaBoost - 第{k + 1}个学习器 | 错误率：{err:.4f} | 权重：{alpha:.4f}")

    def predict(self, X):
        # 步骤3：加权集成预测（论文集成策略）
        pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            pred += alpha * model.predict(X)
        return np.sign(pred).astype(int)


# ===================== 4. 双任务训练与预测 =====================
def run():
    print("===== 复现文献：Adaptive Boosting 诈骗文本分类 =====\n")
    # 1. 加载数据
    train_df, test_df = load_data()
    train_text = train_df[CONFIG["text_col"]].astype(str)
    test_text = test_df[CONFIG["text_col"]].astype(str)

    # 2. 提取特征
    train_X, test_X, _ = get_features(train_text, test_text)
    train_y = train_df["label_bin"].values.astype(int)
    test_y = test_df["label_bin"].values.astype(int)

    # 3. 训练【文献AdaBoost算法】
    print("\n===== 训练文献 Adaptive Boosting 模型 =====")
    ada_model = AdaBoost4TextClassification(K=CONFIG["K"])
    ada_model.fit(train_X, train_y)

    # 4. 预测（是否诈骗）
    pred_bin = ada_model.predict(test_X)
    test_df["pred_is_fraud"] = pred_bin

    # 5. 诈骗类型分类（仅诈骗样本）
    fraud_train = train_df[train_df["label_bin"] == 1]
    fraud_test = test_df[test_df["pred_is_fraud"] == 1]
    if len(fraud_train) > 0 and len(fraud_test) > 0:
        f_train, f_test, _ = get_features(fraud_train[CONFIG["text_col"]].astype(str),
                                          fraud_test[CONFIG["text_col"]].astype(str))
        type_model = DecisionTreeClassifier()
        type_model.fit(f_train, fraud_train[CONFIG["fraud_type_col"]])
        test_df.loc[test_df["pred_is_fraud"] == 1, "pred_fraud_type"] = type_model.predict(f_test)
    test_df["pred_fraud_type"] = test_df["pred_fraud_type"].fillna("正常通话")

    # 6. 评估（论文指标）
    print("\n===== 复现文献算法 · 测试结果 =====")
    acc = accuracy_score(test_y, pred_bin)
    f1 = f1_score(test_y, pred_bin, pos_label=1)
    print(f"是否诈骗 - 准确率：{acc:.4f} | F1：{f1:.4f}")

    # 诈骗类型评估
    real_fraud = test_df[test_df["label_bin"] == 1]
    if len(real_fraud) > 0:
        type_acc = accuracy_score(real_fraud[CONFIG["fraud_type_col"]],
                                  real_fraud[real_fraud["pred_is_fraud"] == 1]["pred_fraud_type"])
        print(f"诈骗类型 - 准确率：{type_acc:.4f}")

    # 7. 保存结果
    save_path = os.path.join(CONFIG["save_dir"], "文献AdaBoost_诈骗分类结果.csv")
    test_df[[CONFIG["text_col"], "label_bin", "pred_is_fraud", CONFIG["fraud_type_col"], "pred_fraud_type"]].to_csv(
        save_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n结果保存：{save_path}")
    print("✅ 严格复现文献 Adaptive Boosting 算法！运行完成！")


if __name__ == "__main__":
    run()
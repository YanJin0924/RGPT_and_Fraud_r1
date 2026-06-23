
# 虚假通话检测与鲁棒性分析（基于 Fraud-R1 多轮诱导策略）

本项目针对电话诈骗文本分类任务，构建了基于BERT与AdaBoost集成（RGPT）的检测模型，并参考Fraud-R1方法设计了6种社会工程诱导策略（信任建立、紧迫感、情感操纵、权威伪装、利益诱惑、模糊性诱导），通过 2 轮和 3 轮多轮语义改写生成增强测试集，系统评估模型在诱导文本下的鲁棒性。项目包含数据处理、增强生成、模型训练、测试评估和可视化全流程。



## 项目结构

```
.
├── 01_data_convert.py               # 原始数据清洗，生成训练/测试 .pkl
├── 02_fraud_r1_augmentation.py      # 基于6种策略生成多轮诱导增强 .pkl
├── 03_Adaboost_fraud_train.py       # RGPT：BERT作为基学习器的AdaBoost训练
├── 04_Adaboost_fraud_test.py        # RGPT集成模型在原始/增强测试集上的预测
├── 05_Adaboost.py                   # 读取性能对比表，绘制DSR和全指标对比图
├── 06_bert_train.py                 # 标准BERT微调，评估并生成性能对比表（Excel）
├── train_dataset/                   # 由01生成：train.pkl, test.pkl
├── dataset/
│   ├── test_clean.csv               # 预处理的测试集明文（需提前准备）
│   └── augmented/                   # 由02生成的增强特征 .pkl 及策略 clean.csv
├── Rgpt_BaseModels/                 # 由03保存的基模型、权重、tokenizer
├── mid_result/                      # 由04生成的各测试集预测结果 .csv（含真值与预测）
├── result/                          # 由06生成的性能对比表 .xlsx，及05生成的图表
└── model/                           # 由06保存的微调BERT模型
```

---

## 环境依赖

- Python 3.8+
- 主要库：
  - `torch` (>=1.10)
  - `transformers` (>=4.20)
  - `scikit-learn` (>=1.0)
  - `pandas`, `numpy`
  - `jieba`
  - `joblib`
  - `openpyxl` (用于Excel读写)
  - `matplotlib` (用于绘图)

安装命令：
```bash
pip install torch transformers scikit-learn pandas numpy jieba joblib openpyxl matplotlib
```

---

## 数据准备

1. 原始数据集（需自行准备）：
   - `训练集结果.csv`：至少包含列 `specific_dialogue_content`（对话文本）和 `is_fraud`（True/False）。
   - `测试集结果.csv`：同上。

2. 预处理的测试集明文：
   - 运行 `06_bert_train.py` 时，它会自动从原始CSV生成 `clean_content` 列，但 `02` 脚本需要依赖 `dataset/test_clean.csv`（包含 `clean_content` 和 `label` 列）。
   - 您需先运行一次 `06_bert_train.py`**（或手动预处理）生成该文件，或直接复制 `06` 中的清洗逻辑生成 `test_clean.csv` 并保存至 `dataset/`。

3. TF-IDF 向量器：
   - `02` 需要加载 `model/tfidf_vectorizer.pkl`。该文件可通过运行 `main.py`（传统AdaBoost）中的 `get_features` 生成后保存，或手动训练并保存。若缺失，可先运行 `main.py`（或单独训练）生成。

---

## 运行步骤（按推荐顺序）

### 1. 数据格式转换（`01_data_convert.py`）
```bash
python 01_data_convert.py
```
- 读取原始CSV，剔除缺失值，提取对话内容与标签，生成 `train_dataset/train.pkl` 和 `test.pkl`。

### 2. 生成增强测试集（`02_fraud_r1_augmentation.py`）
```bash
python 02_fraud_r1_augmentation.py
```
- 依赖 `dataset/test_clean.csv` 和 `model/tfidf_vectorizer.pkl`。
- 对测试集应用6种策略，生成2轮和3轮诱导文本，保存特征向量 `.pkl` 到 `dataset/augmented/`，同时输出 `{策略名}_clean.csv`（供BERT微调评估使用）。

### 3. 训练RGPT集成模型（`03_Adaboost_fraud_train.py`）
```bash
python 03_Adaboost_fraud_train.py
```
- 使用`bert-base-chinese` 作为基学习器，迭代 `K=2` 轮，训练加权集成模型。
- 保存基模型、权重及tokenizer至 `Rgpt_BaseModels/`。

### 4. RGPT集成模型测试（`04_Adaboost_fraud_test.py`）
```bash
python 04_Adaboost_fraud_test.py
```
- 加载训练好的集成模型，对原始测试集和所有增强测试集（6种策略）进行预测。
- 预测结果（`true_label` 和 `pre_label`）保存至 `mid_result/` 目录。

### 5. 标准BERT微调与鲁棒性评估（`06_bert_train.py`）
```bash
python 06_bert_train.py
```
- 微调标准BERT（3 epoch），在原始测试集上评估。
- 读取 `dataset/augmented/` 下的各策略 `_clean.csv`，评估模型在增强数据上的性能。
- 生成 `result/bert_性能对比表.xlsx`，包含 Accuracy、Precision、Recall、F1。
- 若存在 `result/虚假通话检测模型性能对比表.xlsx`（需手动生成或从RGPT结果汇总），则自动合并生成 `AdaBoost_vs_BERT对比.xlsx`。

### 6. 绘制对比图（`05_Adaboost.py`）
```bash
python 05_Adaboost.py
```
- 前提：需先准备好 `result/虚假通话检测模型性能对比表.xlsx`（可从 `04` 的预测结果计算指标得到，或直接使用 `06` 生成的表格改名）。
- 读取该Excel，绘制 DSR防御成功率 柱状图和 DSR/准确率/F1折线图，保存至 `result/figure/`。

---

## 输出结果说明

| 脚本 | 输出文件 | 说明 |
|------|----------|------|
| 01 | `train_dataset/*.pkl` | 训练/测试文本列表及标签 |
| 02 | `dataset/augmented/*.pkl`、`*_clean.csv` | 增强特征向量与明文文本 |
| 03 | `Rgpt_BaseModels/` | 基模型、权重、tokenizer |
| 04 | `mid_result/*.csv` | 各测试集的真值与预测标签 |
| 06 | `result/bert_性能对比表.xlsx`、`bert_*_result.csv` | BERT在各策略下的指标及预测细节 |
| 05 | `result/figure/DSR对比图.png`、`全指标对比图.png` | 可视化图表，可直接用于报告 |

---

## 关键配置

- RGPT 迭代次数：修改 `03` 中的 `K=2`（可调）。
- BERT 微调轮数：修改 `06` 中的 `EPOCHS=3`。
- 文本最大长度：`03` 和 `04` 中的 `MAX_LEN=64`；`06` 中 `max_len=128`。
- 增强策略：可在 `02` 的 `MULTI_ROUND_STRATEGIES` 中自定义或增删。

---

## 注意事项

1. 文件依赖：
   - `02` 需要 `dataset/test_clean.csv` 和 `model/tfidf_vectorizer.pkl`。若缺失，可先运行 `06` 生成 `test_clean.csv`，运行 `main.py`（传统AdaBoost）生成向量器后保存。
   - `05` 需要 `result/虚假通话检测模型性能对比表.xlsx`，该文件并非自动生成。您可以通过以下方式之一准备：
     - 运行 `04` 后，用 `mid_result/*.csv` 手动计算指标并整理成Excel，列名需为：`测试集类型`, `DSR防御成功率`, `准确率`, `F1分数`。
     - 直接使用 `06` 生成的 `bert_性能对比表.xlsx` 重命名并补充DSR列（若需），再运行 `05`。

2. GPU支持：所有脚本自动检测CUDA，若无GPU则使用CPU。

3. 中文显示：`05` 绘图依赖系统字体，若报错可修改 `plt.rcParams['font.sans-serif']` 为系统支持的字体（如 'SimHei' 或 'WenQuanYi Zen Hei'）。

4. 运行顺序灵活性：
   - 若仅需 RGPT 结果，可运行 01→02→03→04，忽略 05 和 06。
   - 若需标准BERT基线，则必须运行06（并确保增强数据已生成）。

---

##  参考文献

- Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements (相关论文)
- AdaBoost 集成学习及BERT预训练模型相关文献

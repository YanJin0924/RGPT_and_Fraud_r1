import pandas as pd
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 
metrics_df = pd.read_excel("result/虚假通话检测模型性能对比表.xlsx")

os.makedirs("result/figure", exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(12, 6))
bars = plt.bar(metrics_df["测试集类型"], metrics_df["DSR防御成功率"], color="#1f77b4", width=0.6)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f"{height:.2%}",
             ha="center", va="bottom", fontsize=10)
plt.title("虚假通话检测模型DSR防御成功率对比", fontsize=14, fontweight="bold")
plt.xlabel("测试集类型", fontsize=12)
plt.ylabel("DSR防御成功率", fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
# 保存图片
plt.savefig("result/figure/DSR对比图.png", dpi=300, bbox_inches="tight")
print(" DSR对比图已保存到：result/figure/DSR对比图.png")

plt.figure(figsize=(14, 7))
metrics = ["DSR防御成功率", "准确率", "F1分数"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, metric in enumerate(metrics):
    plt.plot(metrics_df["测试集类型"], metrics_df[metric], marker="o", color=colors[i], label=metric, linewidth=2)
# 图表设置
plt.title("虚假通话检测模型各指标对比", fontsize=14, fontweight="bold")
plt.xlabel("测试集类型", fontsize=12)
plt.ylabel("指标值", fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.legend(fontsize=11)
plt.grid(linestyle="--", alpha=0.7)
plt.tight_layout()
# 保存图片
plt.savefig("result/figure/全指标对比图.png", dpi=300, bbox_inches="tight")
print("✅ 全指标对比图已保存到：result/figure/全指标对比图.png")

print("\n✅ 所有可视化图表生成完成！可直接插入论文报告")

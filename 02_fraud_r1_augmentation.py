# 强制导入依赖
import jieba
import pandas as pd
import random
import joblib
import os

# 加载数据
test_clean = pd.read_csv("dataset/test_clean.csv")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
os.makedirs("dataset/augmented", exist_ok=True)

MULTI_ROUND_STRATEGIES = {
    "信任建立": [
        "您好，我是官方认证客服，工号{}，全程录音可追溯",
        "我们受监管部门监督，所有操作安全合规，绝不泄露您的隐私",
        "请放心，我会全程协助您处理，绝对保障您的资金安全"
    ],
    "紧迫感": [
        "温馨提示：本次优惠将在24小时后失效，请尽快处理",
        "您的账户存在异常风险，若不及时操作将面临限制使用",
        "最后1小时处理期限，逾期将自动冻结账户，无法恢复"
    ],
    "情感操纵": [
        "我特别理解您的担心，我也是为了帮您避免财产损失",
        "您放心，我会一步步指导您，绝不会让您上当受骗",
        "为了您的资金安全，请一定配合我的操作，别让辛苦钱被骗走"
    ],
    "权威伪装": [
        "您好，我是银行风控专员，检测到您的账户存在异常交易",
        "根据监管要求，您必须立即完成身份核验，否则将冻结账户",
        "这是官方强制流程，请立即按照我的指引完成操作"
    ],
    "利益诱惑": [
        "恭喜您获得专属福利，完成验证即可领取{}元优惠券",
        "本次活动年化收益高达{}%，仅限本次操作可参与",
        "立即完成操作，奖励实时到账，错过永久无法领取"
    ],
    "模糊性诱导": [
        "电话内不便细说，我会把详细资料发送给您，请查收",
        "请添加官方客服微信，我会一对一为您指导操作",
        "点击链接完成核验，所有流程自动处理，无需手动操作"
    ]
}

def multi_round_induce(content, strategy_name, round_num=3):

    content = str(content)
    rounds = MULTI_ROUND_STRATEGIES[strategy_name][:round_num]
    result = content
    for i, line in enumerate(rounds, 1):
        # 随机填充参数
        line = line.format(random.randint(100, 999), random.randint(5, 20))
        result += f" | 第{i}轮诱导：{line}"

    return result


def tokenize(text):
    return " ".join(jieba.lcut(str(text)))

if __name__ == "__main__":
    for strategy in MULTI_ROUND_STRATEGIES.keys():
        aug_2round = test_clean.copy()
        aug_2round["clean_content"] = aug_2round["clean_content"].apply(
            lambda x: multi_round_induce(x, strategy, round_num=2)
        )
        aug_2round["cut_content"] = aug_2round["clean_content"].apply(tokenize)
        X_2round = tfidf.transform(aug_2round["cut_content"]).toarray()
        y_2round = aug_2round["label"].values
        joblib.dump((X_2round, y_2round), f"dataset/augmented/{strategy}_2round.pkl")

        aug_3round = test_clean.copy()
        aug_3round["clean_content"] = aug_3round["clean_content"].apply(
            lambda x: multi_round_induce(x, strategy, round_num=3)
        )
        aug_3round["cut_content"] = aug_3round["clean_content"].apply(tokenize)
        X_3round = tfidf.transform(aug_3round["cut_content"]).toarray()
        y_3round = aug_3round["label"].values
        joblib.dump((X_3round, y_3round), f"dataset/augmented/{strategy}_3round.pkl")

        print(f"{strategy}生成完成：2轮多轮诱导 + 3轮多轮诱导")
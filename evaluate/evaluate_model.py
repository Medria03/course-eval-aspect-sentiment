import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from models.bilstm_attention import MultiTaskAspectSentiment
from preprocess.tokenize import tokenize


def encode(tokens, vocab, max_len: int):
    """
    和训练阶段保持一致的编码逻辑。

    为什么评估时也要重复这一步:
        因为模型接收的是固定长度的 id 序列，
        所以评估数据必须按照与训练完全相同的方式处理。
    """
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, csv_path: Path, vocab, aspect2id, sentiment2id, max_len: int):
        """
        评估阶段使用的数据集类。

        和训练集类类似，只是这里按评估流程组织数据。
        最终仍然返回:
            输入序列、真实主题标签、真实情感标签
        """
        df = pd.read_csv(csv_path)
        self.inputs = []
        self.aspect_ids = []
        self.sentiment_ids = []
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            sentiment = str(row["sentiment"]).strip()
            tokens = tokenize(text)
            self.inputs.append(encode(tokens, vocab, max_len))
            self.aspect_ids.append(aspect2id[aspect])
            self.sentiment_ids.append(sentiment2id[sentiment])

    def __len__(self):
        # 返回评估样本总数。
        return len(self.inputs)

    def __getitem__(self, idx):
        # 按索引取出一条样本，并转成张量。
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.aspect_ids[idx], dtype=torch.long),
            torch.tensor(self.sentiment_ids[idx], dtype=torch.long),
        )


def main():
    """
    评估脚本入口。

    主要流程:
        1. 读取模型元数据
        2. 重建模型结构
        3. 加载模型参数
        4. 在评估数据上逐批预测
        5. 输出分类报告
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "labeled_reviews.csv"))
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "model.pt"))
    parser.add_argument("--meta", default=str(BASE_DIR / "models" / "metadata.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # 读取训练阶段保存的元数据。
    # 没有这些信息，就无法正确重建模型和词表。
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    aspect2id = meta["aspect2id"]
    sentiment2id = meta["sentiment2id"]
    max_len = meta["max_len"]
    embedding_dim = meta["embedding_dim"]
    hidden_dim = meta["hidden_dim"]

    # 构建评估数据集和 DataLoader。
    dataset = ReviewDataset(Path(args.data), vocab, aspect2id, sentiment2id, max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 注意:
    # 这里模型结构必须和训练时完全一致，
    # 否则无法正确加载训练好的参数。
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aspects=len(aspect2id),
        num_sentiments=len(sentiment2id),
    ).to(device)

    # 加载训练好的模型权重。
    model.load_state_dict(torch.load(args.model, map_location=device))

    # 切换到评估模式。
    model.eval()

    # 下面四个列表用来收集:
    # 1. 真实主题标签
    # 2. 预测主题标签
    # 3. 真实情感标签
    # 4. 预测情感标签
    all_aspect_true = []
    all_aspect_pred = []
    all_sent_true = []
    all_sent_pred = []

    # 评估时不需要梯度。
    with torch.no_grad():
        for input_ids, aspect_id, sentiment_id in loader:
            input_ids = input_ids.to(device)
            aspect_logits, sentiment_logits = model(input_ids)

            # 保存真实标签，后面用于和预测结果比较。
            all_aspect_true.extend(aspect_id.tolist())
            all_sent_true.extend(sentiment_id.tolist())

            # argmax 取每条样本得分最高的类别作为预测结果。
            all_aspect_pred.extend(aspect_logits.argmax(dim=1).cpu().tolist())
            all_sent_pred.extend(sentiment_logits.argmax(dim=1).cpu().tolist())

    # 反向构建 id -> 标签名，便于输出可读结果。
    aspect_id2label = {v: k for k, v in aspect2id.items()}
    sentiment_id2label = {v: k for k, v in sentiment2id.items()}

    print("Aspect classification report:")
    # classification_report 会输出:
    # precision: 预测为该类的样本里，有多少是真的
    # recall:    真实属于该类的样本里，有多少被找出来了
    # f1-score:  precision 和 recall 的综合指标
    # support:   该类真实样本数
    print(
        classification_report(
            all_aspect_true,
            all_aspect_pred,
            target_names=[aspect_id2label[i] for i in sorted(aspect_id2label)],
            digits=4,
        )
    )
    print("Sentiment classification report:")
    print(
        classification_report(
            all_sent_true,
            all_sent_pred,
            target_names=[sentiment_id2label[i] for i in sorted(sentiment_id2label)],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()

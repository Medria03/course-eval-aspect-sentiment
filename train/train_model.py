import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from models.bilstm_attention import MultiTaskAspectSentiment
from preprocess.tokenize import tokenize

ASPECT2ID = {
    # 主题标签 -> 数字 id
    # 神经网络不能直接学习中文字符串，
    # 所以要先把主题名映射成整数类别。
    "教学质量": 0,
    "作业负担": 1,
    "课程内容": 2,
    "考试难度": 3,
}

SENTIMENT2ID = {
    # 情感标签 -> 数字 id
    "差评": 0,
    "中评": 1,
    "好评": 2,
}


def build_vocab(texts, min_freq: int = 1):
    """
    根据训练文本构建词表。

    什么是词表:
        词表本质上就是一个字典，记录“词 -> 数字 id”的映射。

    为什么需要词表:
        模型输入必须是数字，不能直接输入中文词语。

    参数:
        texts: 所有训练文本列表。
        min_freq: 最低词频阈值。只有出现次数 >= min_freq 的词才进入词表。
    """
    # 先统计每个词出现了多少次。
    freq = {}
    for text in texts:
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1

    # 先放两个特殊 token:
    # <pad> 用于补齐长度
    # <unk> 用于表示“词表里没有出现过的词”
    vocab = {"<pad>": 0, "<unk>": 1}

    # sorted(freq.items()) 的作用是让词表构建顺序稳定，
    # 这样同一份数据每次构建出来的词表 id 分配都一致。
    for token, count in sorted(freq.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len: int):
    """
    把分词结果编码成固定长度的 id 列表。

    例子:
        tokens = ["老师", "讲课", "很好"]
        可能会变成 [15, 28, 61, 0, 0, 0, ...]

    这里同时完成两件事:
        1. token -> id
        2. 长度统一（不足补 pad，超长截断）
    """
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    # 如果句子太短，就在后面补 <pad>。
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        # 如果句子太长，只保留前 max_len 个 token。
        ids = ids[:max_len]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, csv_path: Path, vocab, aspect2id, sentiment2id, max_len: int):
        """
        自定义数据集类。

        Dataset 的作用:
            告诉 PyTorch“数据长什么样、怎么按索引取一条样本”。

        最终每条样本会被组织成:
            (输入序列张量, 主题标签张量, 情感标签张量)
        """
        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            sentiment = str(row["sentiment"]).strip()

            # 如果数据文件里出现未知标签，直接报错。
            # 这样可以尽早发现标注文件的问题。
            if aspect not in aspect2id:
                raise ValueError(f"Unknown aspect label: {aspect}")
            if sentiment not in sentiment2id:
                raise ValueError(f"Unknown sentiment label: {sentiment}")

            # 文本 -> 分词 -> 固定长度 id 序列
            tokens = tokenize(text)
            input_ids = encode(tokens, vocab, max_len)

            # 最终保存的是纯数字形式，便于后面转张量。
            self.samples.append((input_ids, aspect2id[aspect], sentiment2id[sentiment]))

    def __len__(self):
        # 返回数据集总样本数。
        return len(self.samples)

    def __getitem__(self, idx):
        # 按索引取出一条样本。
        input_ids, aspect_id, sentiment_id = self.samples[idx]
        return (
            # 输入必须是 long 类型，
            # 因为 embedding 查表需要整数索引。
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(aspect_id, dtype=torch.long),
            torch.tensor(sentiment_id, dtype=torch.long),
        )


def train_epoch(model, loader, optimizer, criterion, device):
    """
    训练一个 epoch。

    一个 epoch 的含义:
        把训练集完整地跑一遍。
    """
    # model.train() 会把模型切换到训练模式。
    # 例如 dropout 在训练模式下会生效。
    model.train()
    total_loss = 0.0
    correct_aspect = 0
    correct_sentiment = 0
    total = 0

    # loader 会按 batch 一批一批提供数据。
    for input_ids, aspect_id, sentiment_id in loader:
        # 把数据移动到 CPU 或 GPU。
        input_ids = input_ids.to(device)
        aspect_id = aspect_id.to(device)
        sentiment_id = sentiment_id.to(device)

        # 每个 batch 开始前先清空旧梯度。
        optimizer.zero_grad()

        # 前向传播:
        # 输入文本，得到两个任务的 logits。
        aspect_logits, sentiment_logits = model(input_ids)

        # 分别计算主题损失和情感损失。
        loss_aspect = criterion(aspect_logits, aspect_id)
        loss_sentiment = criterion(sentiment_logits, sentiment_id)

        # 多任务学习里，这里直接把两个损失相加。
        loss = loss_aspect + loss_sentiment

        # 反向传播:
        # 根据 loss 自动计算每个参数的梯度。
        loss.backward()

        # 优化器根据梯度更新参数。
        optimizer.step()

        # 累计总损失。
        # input_ids.size(0) 就是当前 batch 的样本数。
        total_loss += loss.item() * input_ids.size(0)

        # argmax(dim=1) 取每一行分数最大的类别 id。若等于正确答案则正确数+1
        correct_aspect += (aspect_logits.argmax(dim=1) == aspect_id).sum().item()
        correct_sentiment += (sentiment_logits.argmax(dim=1) == sentiment_id).sum().item()
        total += input_ids.size(0)

    # 返回平均损失、主题准确率、情感准确率。
    return total_loss / max(total, 1), correct_aspect / max(total, 1), correct_sentiment / max(total, 1)


def eval_epoch(model, loader, criterion, device):
    """
    在验证集上评估一个 epoch。

    和训练的主要区别:
        1. model.eval()
        2. torch.no_grad()
        3. 不做 backward 和 optimizer.step()
    """
    model.eval()
    total_loss = 0.0
    correct_aspect = 0
    correct_sentiment = 0
    total = 0

    # 关闭梯度计算，节省显存和时间。
    with torch.no_grad():
        for input_ids, aspect_id, sentiment_id in loader:
            input_ids = input_ids.to(device)
            aspect_id = aspect_id.to(device)
            sentiment_id = sentiment_id.to(device)

            aspect_logits, sentiment_logits = model(input_ids)
            loss_aspect = criterion(aspect_logits, aspect_id)
            loss_sentiment = criterion(sentiment_logits, sentiment_id)
            loss = loss_aspect + loss_sentiment

            total_loss += loss.item() * input_ids.size(0)
            correct_aspect += (aspect_logits.argmax(dim=1) == aspect_id).sum().item()
            correct_sentiment += (sentiment_logits.argmax(dim=1) == sentiment_id).sum().item()
            total += input_ids.size(0)

    return total_loss / max(total, 1), correct_aspect / max(total, 1), correct_sentiment / max(total, 1)


def main():
    """
    训练脚本入口。

    运行流程:
        1. 读取命令行参数
        2. 读取数据并构建词表
        3. 构建 Dataset / DataLoader
        4. 初始化模型、损失函数、优化器
        5. 循环训练和验证
        6. 保存模型参数和元数据
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "labeled_reviews.csv"))
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-model", default=str(BASE_DIR / "models" / "model.pt"))
    parser.add_argument("--output-meta", default=str(BASE_DIR / "models" / "metadata.json"))
    args = parser.parse_args()

    # 先读原始标注文件，用于构建词表。
    df = pd.read_csv(args.data)
    vocab = build_vocab(df["text"].tolist())

    # 构建完整数据集对象。
    dataset = ReviewDataset(Path(args.data), vocab, ASPECT2ID, SENTIMENT2ID, args.max_len)
    indices = list(range(len(dataset)))

    # 把样本索引切成训练集和验证集。
    # test_size=0.2 表示 20% 做验证集。
    # random_state=42 表示划分结果可复现。
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # DataLoader 的作用:
    # 按 batch 提供数据，自动把多条样本拼成一个 batch。
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    # 如果有 GPU 就用 GPU，没有就退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化多任务模型。
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_aspects=len(ASPECT2ID),
        num_sentiments=len(SENTIMENT2ID),
    ).to(device)

    # CrossEntropyLoss 是多分类任务最常用的损失函数。
    criterion = nn.CrossEntropyLoss()

    # Adam 是常用优化器，收敛通常比较稳定。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 开始循环训练。
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_a, train_acc_s = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc_a, val_acc_s = eval_epoch(model, val_loader, criterion, device)

        # 每轮打印训练和验证结果，方便观察是否收敛。
        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"aspect_acc={val_acc_a:.2f} sentiment_acc={val_acc_s:.2f}"
        )

    model_path = Path(args.output_model)
    meta_path = Path(args.output_meta)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 只保存模型参数，不保存整个 Python 对象。
    # 这是 PyTorch 中最常见也最稳妥的保存方式。
    torch.save(model.state_dict(), model_path)

    # 推理阶段必须知道:
    # 1. 当时训练用的词表是什么
    # 2. 标签 id 如何映射
    # 3. 模型超参数是什么
    #
    # 所以这里额外保存一份元数据 JSON。
    meta = {
        "vocab": vocab,
        "aspect2id": ASPECT2ID,
        "sentiment2id": SENTIMENT2ID,
        "max_len": args.max_len,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

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
    "教学质量": 0,
    "作业负担": 1,
    "课程内容": 2,
    "考试难度": 3,
}

SENTIMENT2ID = {
    "差评": 0,
    "中评": 1,
    "好评": 2,
}


def build_vocab(texts, min_freq: int = 1):
    # 统计词频并构建词表
    freq = {}
    for text in texts:
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1
    # 约定 <pad>=0, <unk>=1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(freq.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len: int):
    # 将分词结果映射为 id，并做截断/补齐
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, csv_path: Path, vocab, aspect2id, sentiment2id, max_len: int):
        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            sentiment = str(row["sentiment"]).strip()
            if aspect not in aspect2id:
                raise ValueError(f"Unknown aspect label: {aspect}")
            if sentiment not in sentiment2id:
                raise ValueError(f"Unknown sentiment label: {sentiment}")
            tokens = tokenize(text)
            input_ids = encode(tokens, vocab, max_len)
            self.samples.append((input_ids, aspect2id[aspect], sentiment2id[sentiment]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, aspect_id, sentiment_id = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(aspect_id, dtype=torch.long),
            torch.tensor(sentiment_id, dtype=torch.long),
        )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_aspect = 0
    correct_sentiment = 0
    total = 0
    for input_ids, aspect_id, sentiment_id in loader:
        input_ids = input_ids.to(device)
        aspect_id = aspect_id.to(device)
        sentiment_id = sentiment_id.to(device)

        # 前向 + 反向 + 参数更新
        optimizer.zero_grad()
        aspect_logits, sentiment_logits = model(input_ids)
        loss_aspect = criterion(aspect_logits, aspect_id)
        loss_sentiment = criterion(sentiment_logits, sentiment_id)
        loss = loss_aspect + loss_sentiment
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        correct_aspect += (aspect_logits.argmax(dim=1) == aspect_id).sum().item()
        correct_sentiment += (sentiment_logits.argmax(dim=1) == sentiment_id).sum().item()
        total += input_ids.size(0)

    return total_loss / max(total, 1), correct_aspect / max(total, 1), correct_sentiment / max(total, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_aspect = 0
    correct_sentiment = 0
    total = 0
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

    df = pd.read_csv(args.data)
    vocab = build_vocab(df["text"].tolist())

    dataset = ReviewDataset(Path(args.data), vocab, ASPECT2ID, SENTIMENT2ID, args.max_len)
    indices = list(range(len(dataset)))
    # 简单随机划分训练/验证
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 多任务模型：共享编码器 + 双分类头
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_aspects=len(ASPECT2ID),
        num_sentiments=len(SENTIMENT2ID),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_a, train_acc_s = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc_a, val_acc_s = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"aspect_acc={val_acc_a:.2f} sentiment_acc={val_acc_s:.2f}"
        )

    model_path = Path(args.output_model)
    meta_path = Path(args.output_meta)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    # 保存词表与超参，供推理阶段复用
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

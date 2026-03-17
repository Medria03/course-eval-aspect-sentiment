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
    # 将分词映射为 id，并做定长处理
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, csv_path: Path, vocab, aspect2id, sentiment2id, max_len: int):
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
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.aspect_ids[idx], dtype=torch.long),
            torch.tensor(self.sentiment_ids[idx], dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "labeled_reviews.csv"))
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "model.pt"))
    parser.add_argument("--meta", default=str(BASE_DIR / "models" / "metadata.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # 读取训练阶段保存的元数据
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    aspect2id = meta["aspect2id"]
    sentiment2id = meta["sentiment2id"]
    max_len = meta["max_len"]
    embedding_dim = meta["embedding_dim"]
    hidden_dim = meta["hidden_dim"]

    dataset = ReviewDataset(Path(args.data), vocab, aspect2id, sentiment2id, max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aspects=len(aspect2id),
        num_sentiments=len(sentiment2id),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    all_aspect_true = []
    all_aspect_pred = []
    all_sent_true = []
    all_sent_pred = []

    with torch.no_grad():
        for input_ids, aspect_id, sentiment_id in loader:
            input_ids = input_ids.to(device)
            aspect_logits, sentiment_logits = model(input_ids)
            all_aspect_true.extend(aspect_id.tolist())
            all_sent_true.extend(sentiment_id.tolist())
            all_aspect_pred.extend(aspect_logits.argmax(dim=1).cpu().tolist())
            all_sent_pred.extend(sentiment_logits.argmax(dim=1).cpu().tolist())

    aspect_id2label = {v: k for k, v in aspect2id.items()}
    sentiment_id2label = {v: k for k, v in sentiment2id.items()}

    print("Aspect classification report:")
    # classification_report 会输出每个类别的 precision/recall/f1/support
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

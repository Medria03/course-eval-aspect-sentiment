import argparse
import json
import re
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from models.bilstm_attention import MultiTaskAspectSentiment
from preprocess.tokenize import tokenize


def encode(tokens, vocab, max_len: int):
    # 分词 -> id，并做截断/补齐
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def split_clauses(text: str):
    # 用标点进行简单分句，便于一句话多主题输出
    parts = re.split(r"[，,。；;！!？?]+", text)
    return [p.strip() for p in parts if p.strip()]


def predict(text, model, vocab, max_len, id2aspect, id2sentiment, device):
    # 单条文本的主题 + 情感预测
    tokens = tokenize(text)
    input_ids = encode(tokens, vocab, max_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(input_tensor)
    aspect_id = int(aspect_logits.argmax(dim=1).item())
    sentiment_id = int(sentiment_logits.argmax(dim=1).item())
    return id2aspect[aspect_id], id2sentiment[sentiment_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input review text")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "model.pt"))
    parser.add_argument("--meta", default=str(BASE_DIR / "models" / "metadata.json"))
    args = parser.parse_args()

    if not args.text:
        print("Please provide --text for inference.")
        return

    # 读取训练阶段保存的词表与超参
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    aspect2id = meta["aspect2id"]
    sentiment2id = meta["sentiment2id"]
    max_len = meta["max_len"]
    embedding_dim = meta["embedding_dim"]
    hidden_dim = meta["hidden_dim"]

    id2aspect = {v: k for k, v in aspect2id.items()}
    id2sentiment = {v: k for k, v in sentiment2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 按元数据重建模型结构
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aspects=len(aspect2id),
        num_sentiments=len(sentiment2id),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    clauses = split_clauses(args.text)
    if not clauses:
        print("未检测到可分析的文本内容。")
        return

    results = []
    for clause in clauses:
        aspect_label, sentiment_label = predict(
            clause, model, vocab, max_len, id2aspect, id2sentiment, device
        )
        results.append((aspect_label, sentiment_label))

    # Output format: “主题 → 情感”，每行一个结果
    for aspect_label, sentiment_label in results:
        print(f"{aspect_label} → {sentiment_label}")


if __name__ == "__main__":
    main()

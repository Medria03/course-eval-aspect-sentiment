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
    """
    推理阶段的编码函数。

    作用和训练阶段一致:
        把分词结果变成固定长度的数字序列。

    为什么推理也必须这样做:
        因为训练时模型看到的输入就是这种格式，
        推理时如果换了处理方式，模型就无法正确理解输入。
    """
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def split_clauses(text: str):
    """
    用标点做简单分句。

    为什么需要分句:
        因为一条评论中可能同时包含多个主题。

    例子:
        输入: "老师讲课很好，但是作业太多。"
        分句后可能得到:
            ["老师讲课很好", "但是作业太多"]

    当前局限:
        这里只按标点分句，没有做真正的方面抽取，
        所以它是一个简单但直观的工程方案。
    """
    parts = re.split(r"[，,。；;！!？?]+", text)
    return [p.strip() for p in parts if p.strip()]


def predict(text, model, vocab, max_len, id2aspect, id2sentiment, device):
    """
    对一条子句做一次预测。

    输入:
        一条文本子句，例如 "老师讲课很好"

    输出:
        (主题标签, 情感标签)
        例如 ("教学质量", "好评")
    """
    # 先对输入子句进行分词。
    tokens = tokenize(text)

    # 再编码成固定长度 id 序列。
    input_ids = encode(tokens, vocab, max_len)

    # 这里外面再套一层 []，是因为模型希望输入维度是 batch 形式。
    # 即使只预测一句话，也要构造成 [1, seq_len]。
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 推理阶段不需要梯度。
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(input_tensor)

    # 取出得分最高的类别 id。
    aspect_id = int(aspect_logits.argmax(dim=1).item())
    sentiment_id = int(sentiment_logits.argmax(dim=1).item())

    # 把数字 id 映射回中文标签，便于最终展示。
    return id2aspect[aspect_id], id2sentiment[sentiment_id]


def main():
    """
    整个系统的推理入口。

    目标:
        输入一条课程评价，输出若干条“主题 -> 情感”结果。

    总流程:
        1. 读取命令行输入
        2. 加载训练好的模型和元数据
        3. 对原句做分句
        4. 对每个子句分别预测
        5. 输出最终结果
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input review text")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "model.pt"))
    parser.add_argument("--meta", default=str(BASE_DIR / "models" / "metadata.json"))
    args = parser.parse_args()

    if not args.text:
        print("Please provide --text for inference.")
        return

    # 读取训练阶段保存的词表、标签映射和模型超参数。
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    aspect2id = meta["aspect2id"]
    sentiment2id = meta["sentiment2id"]
    max_len = meta["max_len"]
    embedding_dim = meta["embedding_dim"]
    hidden_dim = meta["hidden_dim"]

    # 构建反向字典:
    # 模型预测出的是数字 id，而不是中文标签，
    # 所以后面需要用 id -> 标签名 做反查。
    id2aspect = {v: k for k, v in aspect2id.items()}
    id2sentiment = {v: k for k, v in sentiment2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据训练时保存的超参数，重新搭建同样结构的模型。
    # 只有结构一致，才能正确加载权重。
    model = MultiTaskAspectSentiment(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aspects=len(aspect2id),
        num_sentiments=len(sentiment2id),
    ).to(device)

    # 载入训练好的参数。
    model.load_state_dict(torch.load(args.model, map_location=device))

    # 切换到评估模式。
    model.eval()

    # 先把整句按标点拆成多个子句。
    clauses = split_clauses(args.text)
    if not clauses:
        print("未检测到可分析的文本内容。")
        return

    # 逐个子句预测，并把结果收集起来。
    results = []
    for clause in clauses:
        aspect_label, sentiment_label = predict(
            clause, model, vocab, max_len, id2aspect, id2sentiment, device
        )
        results.append((aspect_label, sentiment_label))

    # 最终输出格式:
    # 每行一条“主题 -> 情感”
    for aspect_label, sentiment_label in results:
        print(f"{aspect_label} → {sentiment_label}")


if __name__ == "__main__":
    main()

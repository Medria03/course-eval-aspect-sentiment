import jieba

from .clean_text import clean_text


def tokenize(text: str):
    # 先清洗再分词，保证输入统一。
    text = clean_text(text)
    if not text:
        return []
    # jieba.lcut 返回分词后的列表，过滤空白 token。
    return [w for w in jieba.lcut(text) if w.strip()]


if __name__ == "__main__":
    sample = "老师讲课很好，但是作业太多。"
    print(tokenize(sample))

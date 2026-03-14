import jieba

from .clean_text import clean_text


def tokenize(text: str):
    text = clean_text(text)
    if not text:
        return []
    return [w for w in jieba.lcut(text) if w.strip()]


if __name__ == "__main__":
    sample = "老师讲课很好，但是作业太多。"
    print(tokenize(sample))

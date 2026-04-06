import jieba

from .clean_text import clean_text


def tokenize(text: str):
    """
    对文本进行“清洗 + 分词”。

    为什么要分词:
        中文句子不像英文那样天然带空格，
        如果不先切成词，模型就很难基于“词”这个粒度学习语义。

    例子:
        输入:  "老师讲课很好，但是作业太多。"
        清洗后: "老师讲课很好但是作业太多"
        分词后: ["老师", "讲课", "很", "好", "但是", "作业", "太", "多"]

    返回:
        一个词列表(list[str])，供后续词表映射和模型训练使用。
    """
    # 第一步: 统一输入格式。
    # 这里会调用 clean_text，把英文、数字、标点等先去掉。
    text = clean_text(text)
    if not text:
        # 如果清洗后为空，就返回空列表。
        # 后续 encode 会把空列表补成全 pad 序列。
        return []
    # 第二步: 使用 jieba 进行中文分词。
    # jieba.lcut 会直接返回 Python 列表，而不是生成器，便于后续直接使用。
    #
    # w.strip() 的作用:
    # 去掉可能出现的空白字符，如果某个 token 清洗后是空串，就过滤掉。
    return [w for w in jieba.lcut(text) if w.strip()]


if __name__ == "__main__":
    sample = "老师讲课很好，但是作业太多。"
    print(tokenize(sample))

import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 只保留中文字符，去掉标点/数字/英文等噪声。
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


if __name__ == "__main__":
    sample = "老师讲课很好，但是作业太多!!!"
    print(clean_text(sample))

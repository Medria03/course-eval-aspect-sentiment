import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Keep only Chinese characters to match typical course reviews.
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


if __name__ == "__main__":
    sample = "老师讲课很好，但是作业太多!!!"
    print(clean_text(sample))

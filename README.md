# 基于NLP的学生课程评价主题-情感联合分析系统

本项目实现“课程评价的主题识别 + 情感判断”的联合分析流程，支持从原始评论到训练、评估与推理的完整链路。

## 功能概览

- 主题(Aspect)分类：如“教学质量 / 作业负担 / 课程内容 / 考试难度”
- 情感(Sentiment)分类：差评/中评/好评
- 模型结构：Embedding + BiLSTM + Attention + 双头输出(主题/情感)

## 目录结构

```
course_eval_nlp/
├── data/
│   ├── raw_reviews.csv          # 原始评论(未标注)
│   └── labeled_reviews.csv      # 标注数据集
├── preprocess/
│   ├── clean_text.py            # 清洗文本
│   └── tokenize.py              # 分词
├── models/
│   └── bilstm_attention.py      # 模型定义
├── train/
│   └── train_model.py           # 训练脚本
├── evaluate/
│   └── evaluate_model.py        # 评估脚本
├── main.py                      # 推理入口
└── requirements.txt             # 依赖列表
```

## 环境依赖

- Python 3.8+
- PyTorch
- pandas
- jieba
- scikit-learn

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集格式

`data/labeled_reviews.csv` 需要包含以下三列：

| 字段 | 含义 |
| --- | --- |
| text | 评论文本 |
| aspect | 主题标签 |
| sentiment | 情感标签 |

示例：

```
text,aspect,sentiment
老师讲课很好,教学质量,好评
作业太多了,作业负担,差评
课程内容一般,课程内容,中评
考试有点难,考试难度,中评
```

### 标签编码

情感标签编码：

- 差评 = 0
- 中评 = 1
- 好评 = 2

主题标签编码：

- 教学质量 = 0
- 作业负担 = 1
- 课程内容 = 2
- 考试难度 = 3

编码在 `train/train_model.py` 中配置，如需扩展请同步修改。

## 文本预处理

流程：

原始文本 → 清洗 → 分词 → 去停用词(可选)

当前实现：

- `clean_text.py` 仅保留中文字符
- `tokenize.py` 使用 jieba 分词

如需停用词，可在 `tokenize.py` 中加入停用词过滤逻辑。

## 训练模型

运行：

```bash
python train/train_model.py --epochs 5
```

常用参数：

- `--data` 标注数据路径
- `--embedding-dim` 词向量维度
- `--hidden-dim` LSTM隐层维度
- `--max-len` 最大序列长度
- `--batch-size` 批大小
- `--epochs` 训练轮数
- `--lr` 学习率
- `--output-model` 模型保存路径
- `--output-meta` 元数据保存路径

输出：

- `models/model.pt`
- `models/metadata.json`

## 评估模型

运行：

```bash
python evaluate/evaluate_model.py
```

输出：

- 主题分类指标 (precision/recall/f1)
- 情感分类指标 (precision/recall/f1)

## 推理(系统演示)

运行：

```bash
python main.py --text "老师讲课很好，但是作业太多"
```

示例输出：

```
老师讲课很好 -> 教学质量 / 好评
但是作业太多 -> 作业负担 / 差评
```

说明：

- 推理时会对输入文本做简单分句(逗号/句号/感叹号等)
- 每个子句进行主题+情感预测

## 建议扩展方向

- 增加“方面抽取”模块，替换简单分句
- 使用更强的预训练模型(BERT/ERNIE)
- 扩展主题类别与情感等级
- 加入可视化界面或Web接口

## 快速检查清单

- 已准备标注数据(>=1000条为佳)
- 运行训练脚本完成模型训练
- 运行评估脚本查看指标
- 运行推理脚本验证效果

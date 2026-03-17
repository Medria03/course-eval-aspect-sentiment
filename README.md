# 基于NLP的学生课程评价主题-情感联合分析系统

面向本科毕业设计的完整实现方案，覆盖数据构建、预处理、模型训练、评估与推理演示。

## 研究背景与目标

学生课程评价包含“评价对象(主题)”与“情感倾向”。本项目目标是实现一个可复现、可扩展的联合分析系统，完成以下任务：

- 主题识别(Aspect)：如“教学质量 / 作业负担 / 课程内容 / 考试难度”
- 情感判断(Sentiment)：差评 / 中评 / 好评

## 系统流程

```
输入评论
  ↓
主题识别 (Aspect)
  ↓
情感判断 (Sentiment)
  ↓
输出: 主题 → 情感
```

## 项目结构

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

## 数据集构建与标注规范

推荐至少 1000 条标注评论，保证主题与情感分布尽量均衡。

标注原则：

- 每行只包含一个主题
- 一条评论包含多个主题时，拆成多行
- 情感标签只在该主题范围内判定

格式：`data/labeled_reviews.csv`

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

## 预处理流程

- 清洗：仅保留中文字符
- 分词：jieba 分词
- 去停用词：可选(可在 `tokenize.py` 中加入)

## 模型设计

采用“共享编码器 + 双任务输出”的联合建模方案：

- Embedding 将词映射为向量
- BiLSTM 获取上下文表示
- Attention 聚合关键特征
- 双头输出：Aspect 分类、Sentiment 分类

模型定义在 `models/bilstm_attention.py`。

## 训练与评估

训练：

```bash
python train/train_model.py --epochs 5
```

评估：

```bash
python evaluate/evaluate_model.py
```

输出指标：

- Accuracy
- Precision
- Recall
- F1-score

## 推理示例

运行：

```bash
python main.py --text "老师讲课很好，但是作业太多"
```

输出：

```
教学质量 → 好评
作业负担 → 差评
```

说明：

- 推理时对输入文本进行分句，再逐句预测
- 输出格式为“主题 → 情感”，不显示原子句

## 复现实验步骤

1. 准备 `data/labeled_reviews.csv`
2. 安装依赖 `pip install -r requirements.txt`
3. 训练模型 `python train/train_model.py`
4. 评估模型 `python evaluate/evaluate_model.py`
5. 推理测试 `python main.py --text "..."`

## 可扩展方向

- 引入 BERT/ERNIE 等预训练模型
- 增加方面抽取模块(替代简单分句)
- 设计 Web/UI 演示界面
- 增加多标签或细粒度情感等级

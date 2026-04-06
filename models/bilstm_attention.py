import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        """
        文本编码器: Embedding + BiLSTM + Attention。

        这个类只做一件事:
        把一句已经编码成 id 的文本，变成一个“整句向量”。

        参数:
            vocab_size: 词表大小。词表里一共有多少个不同 token。
            embedding_dim: 每个词映射成多少维向量。
            hidden_dim: LSTM 单个方向上的隐层维度。
            dropout: dropout 比例，用来减轻过拟合。

        最终输出:
            一个形状为 [batch_size, hidden_dim * 2] 的句向量。
            这里乘 2 是因为用了双向 LSTM。
        """
        super().__init__()
        # Embedding 层作用:
        # 把“词 id”转换成“词向量”。
        #
        # 例如:
        # 输入是 [12, 35, 8]
        # 输出就会变成 3 个 embedding_dim 维的向量。
        #
        # padding_idx=0 的含义:
        # 如果某个位置的 id 是 0（也就是 <pad>），
        # 这个位置对应的向量不会像普通词一样被更新。

        # 我们要准备一个有 vocab_size 行、每行有 embedding_dim 个数字的表格。
        # 当你输入词 ID（索引）时，它就去那一行把那一串数字“抓”出来，作为这个词的特征表示。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 层作用:
        # 让模型按时间顺序“读句子”，学习上下文信息。
        #
        # batch_first=True:
        # 输入张量形状采用 [batch, seq_len, feature]，
        # 对新手更直观。
        #
        # bidirectional=True:
        # 双向 LSTM，既看左边上下文，也看右边上下文。
        # 所以最终每个时间步的输出维度是 hidden_dim * 2。
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Attention 打分层:
        # 对每个时间步输出一个分数，表示“这个位置有多重要”。
        # 这里只是最简单的线性 attention。
        # 物理意义：它给 BiLSTM 输出的每一个时间步（每个词）的特征向量做一个加权映射。
        # 你代码的逻辑：你的代码里没有 $s_t$（因为不是翻译任务，没有 Decoder）。
        # 于是你用一个会学习的线性层 nn.Linear 来代替这种“询问”。套用解释：这个 nn.Linear 
        # 就像是一个**“资深考官”**。BiLSTM 输出的每个词向量 $h_i$ 都要经过这个考官的面试。
        # 考官心里有一套标准（权重 $W$），给每个词打分 $e_i$。

        # 深度解释： 当你写下这行代码时，你实际上在内存里申请了一块空间，并放入了一个随机的矩阵 $W$。
        # 我并没有人工干预每个词的重要性，而是通过 nn.Linear 构造了一个待学习的注意力空间，
        # 让模型在处理成千上万条评价后，自动收敛出最能代表‘重要性’的权重矩阵。”
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Dropout:
        # 训练时随机丢弃一部分特征，减少过拟合。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播流程。

        参数:
            x: 形状为 [batch_size, seq_len] 的整数张量，
               每个元素都是词表中的 token id。

        返回:
            context: 形状为 [batch_size, hidden_dim * 2] 的句向量。
        """
        # x 的每一行是一句话，每个数字代表一个词。
        # 例如:
        # [
        #   [15, 21,  8, 0, 0],
        #   [ 4, 99, 31, 7, 2]
        # ]
        embed = self.embedding(x)

        # embed 的形状:
        # [batch_size, seq_len, embedding_dim]
        #
        # 可以理解为:
        # “原来每个位置只是一个数字 id，
        #  现在每个位置都变成了一个向量表示。”
        lstm_out, _ = self.lstm(embed)

        # lstm_out 的形状:
        # [batch_size, seq_len, hidden_dim * 2]
        #
        # 含义:
        # 每个时间步都有一个包含上下文信息的表示。
        attn_scores = self.attention(lstm_out).squeeze(-1)

        # attn_scores 的形状:
        # [batch_size, seq_len]
        #
        # attention 线性层先对每个时间步给一个“原始分数”。
        # squeeze(-1) 是因为线性层输出最后一维大小为 1，
        # 这里把那一维去掉，让张量更方便做 softmax。
        attn_weights = torch.softmax(attn_scores, dim=1)

        # softmax 后，attention 权重在 seq_len 维度上相加等于 1。
        # 可以理解为:
        # “模型认为一句话里每个词分别有多重要”。

        # attn_weights.unsqueeze(-1) 的作用:
        # 把形状从 [batch, seq_len] 变成 [batch, seq_len, 1]，
        # 这样才能与 lstm_out 做逐位置相乘。
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        # 加权求和后得到整句表示:
        # [batch_size, hidden_dim * 2]
        #
        # 这一步相当于把一句话压缩成一个向量，
        # 后面的分类层就基于这个向量做判断。
        return self.dropout(context)


class MultiTaskAspectSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_aspects: int,
        num_sentiments: int,
        dropout: float = 0.2,
    ):
        """
        多任务模型:
        一个共享编码器，同时做两个分类任务。

        任务 1: 主题分类 (Aspect)
        任务 2: 情感分类 (Sentiment)

        为什么要共享编码器:
            因为这两个任务都需要先理解句子语义，
            共享底层表示可以减少参数量，也能让两个任务互相提供信息。
        """
        super().__init__()

        # 共享的文本编码器。
        self.encoder = BiLSTMAttention(vocab_size, embedding_dim, hidden_dim, dropout=dropout)

        # 主题分类头:
        # 输入是句向量，输出是“每个主题类别的分数”。
        self.aspect_head = nn.Linear(hidden_dim * 2, num_aspects)

        # 情感分类头:
        # 输入仍然是同一个句向量，输出是“每个情感类别的分数”。
        self.sentiment_head = nn.Linear(hidden_dim * 2, num_sentiments)

    def forward(self, x: torch.Tensor):
        """
        返回两个输出:
            aspect_logits: 主题分类分数
            sentiment_logits: 情感分类分数

        注意:
            这里返回的是 logits，不是最终标签。
            真正的类别需要在后面用 argmax 取最大分数的位置。
        """
        context = self.encoder(x)
        aspect_logits = self.aspect_head(context)
        sentiment_logits = self.sentiment_head(context)
        return aspect_logits, sentiment_logits

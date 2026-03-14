import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
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
        super().__init__()
        self.encoder = BiLSTMAttention(vocab_size, embedding_dim, hidden_dim, dropout=dropout)
        self.aspect_head = nn.Linear(hidden_dim * 2, num_aspects)
        self.sentiment_head = nn.Linear(hidden_dim * 2, num_sentiments)

    def forward(self, x: torch.Tensor):
        context = self.encoder(x)
        aspect_logits = self.aspect_head(context)
        sentiment_logits = self.sentiment_head(context)
        return aspect_logits, sentiment_logits

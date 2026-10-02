"""Two-axis residual vision classifier built only from standard tanh RNNs and linear layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

ARCHITECTURES = {
    "RNN2D-ARCH-S": {"embed_dim": 192, "rnn_hidden_size": 48, "num_blocks": 2},
    "RNN2D-ARCH-M": {"embed_dim": 256, "rnn_hidden_size": 64, "num_blocks": 3},
    "RNN2D-ARCH-L": {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3},
}
ARCHITECTURE_IDS = tuple(ARCHITECTURES)


def horizontal_sequences(x: Tensor) -> Tensor:
    """Group image rows; timestep index is the original column index."""
    if x.ndim != 4 or x.shape[1:3] != (7, 7):
        raise ValueError(f"Expected [B,7,7,C], got {tuple(x.shape)}")
    return x.contiguous().reshape(x.shape[0] * 7, 7, x.shape[3])


def vertical_sequences(x: Tensor) -> Tensor:
    """Group image columns; timestep index is the original row index."""
    if x.ndim != 4 or x.shape[1:3] != (7, 7):
        raise ValueError(f"Expected [B,7,7,C], got {tuple(x.shape)}")
    return x.permute(0, 2, 1, 3).contiguous().reshape(x.shape[0] * 7, 7, x.shape[3])


def restore_vertical(output: Tensor, batch_size: int) -> Tensor:
    if output.ndim != 3 or output.shape[:2] != (batch_size * 7, 7):
        raise ValueError(f"Unexpected vertical RNN output: {tuple(output.shape)}")
    return output.reshape(batch_size, 7, 7, output.shape[-1]).permute(0, 2, 1, 3).contiguous()


class TokenProjection(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(190, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (190, 7, 7):
            raise ValueError(f"Expected [B,190,7,7], got {tuple(x.shape)}")
        tokens = x.permute(0, 2, 3, 1).contiguous()
        result = self.norm(self.linear(tokens))
        if result.shape != (x.shape[0], 7, 7, self.linear.out_features):
            raise AssertionError("Token projection shape changed")
        return result


class BiRNN2DBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int, dropout_probability: float = 0.0) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.horizontal_rnn = nn.RNN(embed_dim, hidden_size, num_layers=1,
                                     nonlinearity="tanh", batch_first=True,
                                     bidirectional=True, dropout=0.0)
        self.vertical_rnn = nn.RNN(embed_dim, hidden_size, num_layers=1,
                                   nonlinearity="tanh", batch_first=True,
                                   bidirectional=True, dropout=0.0)
        self.fusion = nn.Linear(4 * hidden_size, embed_dim)
        self.fusion_dropout = nn.Dropout(dropout_probability) if dropout_probability else nn.Identity()
        self.channel_norm = nn.LayerNorm(embed_dim)
        mlp_layers: list[nn.Module] = [nn.Linear(embed_dim, 2 * embed_dim), nn.ReLU()]
        if dropout_probability:
            mlp_layers.append(nn.Dropout(dropout_probability))
        mlp_layers.append(nn.Linear(2 * embed_dim, embed_dim))
        if dropout_probability:
            mlp_layers.append(nn.Dropout(dropout_probability))
        self.channel_mlp = nn.Sequential(*mlp_layers)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (7, 7, self.embed_dim):
            raise ValueError(f"Expected [B,7,7,{self.embed_dim}], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        y = self.pre_norm(x)
        horizontal, _ = self.horizontal_rnn(horizontal_sequences(y))
        vertical, _ = self.vertical_rnn(vertical_sequences(y))
        if (horizontal.shape != (batch_size * 7, 7, 2 * self.hidden_size)
                or vertical.shape != horizontal.shape):
            raise AssertionError("Two-axis recurrent output shape changed")
        horizontal = horizontal.reshape(batch_size, 7, 7, 2 * self.hidden_size)
        vertical = restore_vertical(vertical, batch_size)
        fused_axes = torch.cat((horizontal, vertical), dim=-1)
        if fused_axes.shape != (batch_size, 7, 7, 4 * self.hidden_size):
            raise AssertionError("Axis fusion shape changed")
        x = x + self.fusion_dropout(self.fusion(fused_axes))
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class VisionRNN2DClassifier(nn.Module):
    def __init__(self, architecture_id: str, dropout_probability: float = 0.0) -> None:
        super().__init__()
        if architecture_id not in ARCHITECTURES:
            raise ValueError(f"Unknown two-axis RNN architecture: {architecture_id}")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Dropout probability must be in [0,1)")
        self.architecture_id = architecture_id
        self.specification = ARCHITECTURES[architecture_id]
        embed_dim = self.specification["embed_dim"]
        self.projection = TokenProjection(embed_dim)
        self.blocks = nn.ModuleList(BiRNN2DBlock(embed_dim, self.specification["rnn_hidden_size"],
                                                 dropout_probability)
                                    for _ in range(self.specification["num_blocks"]))
        self.head_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.projection(x)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=(1, 2))
        if pooled.shape != (x.shape[0], self.specification["embed_dim"]):
            raise AssertionError("Global spatial pooling shape changed")
        logits = self.classifier(self.head_norm(pooled))
        if logits.shape != (x.shape[0], 2):
            raise AssertionError("RNN2D classifier output shape changed")
        return logits

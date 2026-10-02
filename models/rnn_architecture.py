"""Three standard-RNN classifiers for the frozen 190-channel spatial feature map."""

from __future__ import annotations

import torch
from torch import Tensor, nn

ARCHITECTURE_IDS = ("RNN-ARCH-A", "RNN-ARCH-B", "RNN-ARCH-C")
SPECIFICATIONS = {
    "RNN-ARCH-A": {"sequence_representation": "row", "sequence_length": 7,
                   "rnn_input_size": 1330, "hidden_size": 128, "num_layers": 1, "bidirectional": False},
    "RNN-ARCH-B": {"sequence_representation": "row", "sequence_length": 7,
                   "rnn_input_size": 1330, "hidden_size": 96, "num_layers": 1, "bidirectional": True},
    "RNN-ARCH-C": {"sequence_representation": "cell_raster", "sequence_length": 49,
                   "rnn_input_size": 190, "hidden_size": 128, "num_layers": 1, "bidirectional": True},
}


def _check_map(x: Tensor) -> None:
    if x.ndim != 4 or x.shape[1:] != (190, 7, 7):
        raise ValueError(f"Expected [B,190,7,7], got {tuple(x.shape)}")


def feature_map_to_row_sequence(x: Tensor) -> Tensor:
    """Top-to-bottom rows; each timestep holds all channels and columns."""
    _check_map(x)
    return x.permute(0, 2, 1, 3).contiguous().reshape(x.shape[0], 7, 1330)


def feature_map_to_cell_sequence(x: Tensor) -> Tensor:
    """Row-major raster order, each timestep holding the 190 channel values."""
    _check_map(x)
    return x.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], 49, 190)


class ArchitectureRNN(nn.Module):
    def __init__(self, architecture_id: str) -> None:
        super().__init__()
        if architecture_id not in SPECIFICATIONS:
            raise ValueError(f"Unknown RNN architecture: {architecture_id}")
        self.architecture_id = architecture_id
        self.specification = SPECIFICATIONS[architecture_id]
        spec = self.specification
        self.rnn = nn.RNN(input_size=spec["rnn_input_size"], hidden_size=spec["hidden_size"],
                          num_layers=1, nonlinearity="tanh", batch_first=True,
                          bidirectional=spec["bidirectional"], dropout=0.0)
        self.classifier = nn.Linear(spec["hidden_size"] * (2 if spec["bidirectional"] else 1), 2)

    def forward(self, x: Tensor) -> Tensor:
        sequence = (feature_map_to_cell_sequence(x) if self.architecture_id == "RNN-ARCH-C"
                    else feature_map_to_row_sequence(x))
        _, h_n = self.rnn(sequence)
        # For bidirectional RNNs the last two hidden states are the final
        # forward and backward states; output[:, -1, :] is not equivalent.
        representation = (torch.cat((h_n[-2], h_n[-1]), dim=1)
                          if self.specification["bidirectional"] else h_n[-1])
        return self.classifier(representation)

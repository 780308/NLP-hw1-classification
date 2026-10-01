"""A compact fully connected baseline for RGB cat/dog images."""

from torch import Tensor, nn


class DNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_dims: tuple[int, int] = (256, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_features = 3 * input_size * input_size
        first_hidden, second_hidden = hidden_dims
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 2),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)

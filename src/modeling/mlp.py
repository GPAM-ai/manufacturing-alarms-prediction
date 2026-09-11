"""PyTorch MLP architecture for daily alarm-recurrence regression."""

from collections.abc import Sequence

from torch import Tensor, nn


class MLPRegressor(nn.Module):
    """Predict one continuous value from each row of engineered features.

    Input tensors must follow the tabular convention ``(samples, features)``.
    Every hidden layer applies a linear transformation, ReLU activation, and
    dropout. The final linear layer returns one unconstrained regression value
    per sample.

    Args:
        input_size: Number of predictors in the selected feature dataset.
        hidden_sizes: Number of units in each hidden layer, in order.
        dropout: Dropout probability applied after every hidden activation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if (
            not isinstance(input_size, int)
            or isinstance(input_size, bool)
            or input_size <= 0
        ):
            raise ValueError('input_size must be a positive integer')
        if isinstance(hidden_sizes, (str, bytes)):
            raise TypeError('hidden_sizes must be a sequence of positive integers')

        hidden_sizes = tuple(hidden_sizes)
        if not hidden_sizes:
            raise ValueError('hidden_sizes must contain at least one hidden layer')
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in hidden_sizes
        ):
            raise ValueError('hidden_sizes must contain only positive integers')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in the interval [0, 1)')

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend((
                nn.Linear(previous_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one regression prediction for every feature row."""
        if inputs.ndim != 2:
            raise ValueError(
                'inputs must have shape (samples, features); '
                f'received {tuple(inputs.shape)}'
            )
        if inputs.size(1) != self.input_size:
            raise ValueError(
                f'expected {self.input_size} features per sample, '
                f'received {inputs.size(1)}'
            )

        return self.network(inputs).squeeze(-1)


__all__ = ['MLPRegressor']

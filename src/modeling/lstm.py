"""PyTorch LSTM architecture for daily alarm-recurrence regression."""

from torch import Tensor, nn


class LSTMRegressor(nn.Module):
    """Predict one continuous value from each multivariate temporal window.

    Input tensors must follow the project convention
    ``(samples, time_steps, features)``. The last LSTM output summarizes the
    complete input window and is passed to a small nonlinear regression head.

    Args:
        input_size: Number of input features at each time step.
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of stacked recurrent layers.
        dense_size: Number of units in the regression head's hidden layer.
        dropout: Dropout probability. For stacked LSTMs it is also applied
            between recurrent layers; it is always applied in the regression
            head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dense_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        for name, value in (
            ('input_size', input_size),
            ('hidden_size', hidden_size),
            ('num_layers', num_layers),
            ('dense_size', dense_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in the interval [0, 1)')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dense_size = dense_size
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one regression prediction for every input sequence."""
        if inputs.ndim != 3:
            raise ValueError(
                'inputs must have shape (samples, time_steps, features); '
                f'received {tuple(inputs.shape)}'
            )
        if inputs.size(1) == 0:
            raise ValueError('inputs must contain at least one time step')
        if inputs.size(2) != self.input_size:
            raise ValueError(
                f'expected {self.input_size} features per time step, '
                f'received {inputs.size(2)}'
            )

        sequence_output, _ = self.lstm(inputs)
        last_output = sequence_output[:, -1, :]
        return self.regression_head(last_output).squeeze(-1)


__all__ = ['LSTMRegressor']

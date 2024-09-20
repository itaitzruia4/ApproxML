import numpy as np
from sklearn.base import BaseEstimator


def relu(x: np.ndarray):
    return x * (x > 0)


class ELM(BaseEstimator):
    """
    Extreme Learning Machine (ELM) implementation.
    """

    def __init__(self, input_size, hidden_size):
        # Generate the weight vector and bias at random.
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.input_weights = np.random.rand(input_size, hidden_size)
        self.hidden_biases = np.random.rand(hidden_size)
        self.output_weights = None

    def _calculate_hidden_output(self, X):
        W, b = self.input_weights, self.hidden_biases

        # Calculate the hidden layer output matrix H.
        B = np.full((X.shape[0], b.shape[0]), b)
        hidden_output = relu(np.dot(X, W) + B)

        ones_col = np.ones((hidden_output.shape[0], 1))
        H = np.concatenate([ones_col, hidden_output], axis=1)
        return H

    def fit(self, X, y, sample_weight=None):
        # Estimate the output weight β∗ = H†T.
        H = self._calculate_hidden_output(X)

        if sample_weight is not None:
            H = H * np.sqrt(sample_weight[:, None])
            y = y * np.sqrt(sample_weight)

        H_pinv = np.linalg.pinv(H)
        output_weights = H_pinv @ y
        self.output_weights = output_weights

    def predict(self, X):
        H = self._calculate_hidden_output(X)
        return np.dot(H, self.output_weights)

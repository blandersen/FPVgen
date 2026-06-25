"""
flamelet_nn.py
-----------------
Defines the neural network architecture and the PyTorch Dataset
that wraps flamelet data.

Inputs  : an ordered list of control variables, any of
          Z  (mixture fraction, already in [0,1])
          Zv (mixture fraction variance, normalised by Z*(1-Z))
          C  (progress variable)
          The default raw-flamelet workflow uses (Z, C); Zv is supported but is
          identically zero for laminar flamelet data.
Outputs : T, Y_k (k species), rho, mu, lambda, cp, omega_dot, ...
"""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ── 1.  Dataset ────────────────────────────────────────────────────────────────

class FlametableDataset(Dataset):
    """
    Wraps pre-normalised numpy arrays X (N, n_inputs) and Y (N, M).
    Call FlametableDataset.from_raw(...) to normalise from physical data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 Y_min: np.ndarray, Y_max: np.ndarray):
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.Y    = torch.tensor(Y, dtype=torch.float32)
        self.Y_min = Y_min          # kept for inverse-transform at inference
        self.Y_max = Y_max

    # ── normalisation ──────────────────────────────────────────────────────────

    @staticmethod
    def normalise_inputs(inputs: dict, input_names: Sequence[str]) -> np.ndarray:
        """Build the normalised input matrix from named control-variable arrays.

        Args:
            inputs: Mapping of control-variable name -> 1-D array (N,). Must
                contain every name in ``input_names``. ``Z`` is assumed already
                in [0, 1]; ``Zv`` (if present) is treated as a physical variance
                and divided by Z*(1-Z) to give a normalised value in [0, 1];
                all other variables are passed through unchanged.
            input_names: Ordered list of control variables to stack as columns.

        Returns:
            X of shape (N, len(input_names)).
        """
        eps = 1e-10
        cols = []
        for name in input_names:
            if name not in inputs:
                raise KeyError(f"Input '{name}' not found in provided data: {list(inputs)}")
            arr = np.asarray(inputs[name], dtype=np.float64)
            if name == "Zv":
                Z = np.asarray(inputs["Z"], dtype=np.float64)
                arr = np.clip(arr / (Z * (1.0 - Z) + eps), 0.0, 1.0)
            cols.append(arr)
        return np.stack(cols, axis=1)

    @staticmethod
    def normalise_outputs(Y: np.ndarray):
        """Min-max scales each output column to [0, 1]."""
        eps   = 1e-10
        Y_min = Y.min(axis=0)
        Y_max = Y.max(axis=0)
        Y_n   = (Y - Y_min) / (Y_max - Y_min + eps)
        return Y_n, Y_min, Y_max

    @staticmethod
    def denormalise_outputs(Y_n: np.ndarray,
                            Y_min: np.ndarray,
                            Y_max: np.ndarray) -> np.ndarray:
        return Y_n * (Y_max - Y_min) + Y_min

    # ── convenience constructor ────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, inputs: dict, input_names: Sequence[str], Y: np.ndarray):
        """
        inputs      : mapping of name -> 1-D numpy array (N,)
        input_names : ordered list of control variables to use as inputs
        Y           : 2-D numpy array (N, M)
        """
        X   = cls.normalise_inputs(inputs, input_names)
        Y_n, Y_min, Y_max = cls.normalise_outputs(Y)
        return cls(X, Y_n, Y_min, Y_max)

    # ── PyTorch Dataset interface ──────────────────────────────────────────────

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── 2.  Model ──────────────────────────────────────────────────────────────────

class FlametableNN(nn.Module):
    """
    Fully-connected MLP with tanh activations and a linear output layer.

    Default: 4 hidden layers × 128 neurons  (≈ 70 K parameters for 25 outputs)
    Increase hidden_dim to 256 for larger mechanisms or more outputs.

    Args
    ----
    n_inputs    : number of control variables (default 3: Z, Zv, C)
    n_outputs   : number of tabulated quantities (T + n_species + transport + ...)
    hidden_dim  : neurons per hidden layer
    n_layers    : number of hidden layers
    """

    def __init__(self,
                 n_inputs:   int = 3,
                 n_outputs:  int = 25,
                 hidden_dim: int = 128,
                 n_layers:   int = 4):
        super().__init__()

        layers: list[nn.Module] = []

        # input → first hidden
        layers += [nn.Linear(n_inputs, hidden_dim), nn.Tanh()]

        # hidden → hidden
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]

        # last hidden → output  (linear, no activation)
        layers.append(nn.Linear(hidden_dim, n_outputs))

        self.net = nn.Sequential(*layers)

        # store config so we can reconstruct the model from a checkpoint
        self.config = dict(n_inputs=n_inputs, n_outputs=n_outputs,
                           hidden_dim=hidden_dim, n_layers=n_layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init — sensible default for tanh networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
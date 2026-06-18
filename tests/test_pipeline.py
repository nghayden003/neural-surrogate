"""
test_pipeline.py
================

Minimal tests for the surrogate pipeline.

We do not unit-test a model's accuracy (that is what the evaluation is for). 
We test that data has the right shapes, that transforms invert correctly, 
and that a forward pass produces the expected output dimension.

Run:
    pytest
"""

import numpy as np
import torch

from src.data import (
    generate_dataset, split_dataset, compute_stats, normalise, denormalise,
)
from src.model import MLPSurrogate


def test_dataset_shapes():
    """Generated data has expected shapes."""
    X, Y, x_grid = generate_dataset(n_samples=20, n_points=32, seed=1)
    assert X.shape == (20, 4)
    assert Y.shape == (20, 32)
    assert x_grid.shape == (32,)


def test_split_partitions_all_data():
    """Train/val/test split keeps every row exactly once."""
    X, Y, _ = generate_dataset(n_samples=40, n_points=16, seed=2)
    (Xtr, _), (Xva, _), (Xte, _) = split_dataset(X, Y, seed=2)
    assert Xtr.shape[0] + Xva.shape[0] + Xte.shape[0] == 40


def test_normalisation_inverts():
    """denormalise(normalise(x)) recovers the original array."""
    X, _, _ = generate_dataset(n_samples=30, n_points=16, seed=3)
    mean, std = compute_stats(X)
    recovered = denormalise(normalise(X, mean, std), mean, std)
    assert np.allclose(recovered, X, atol=1e-5)


def test_forward_pass_output_dim():
    """The model maps a batch of parameter vectors to the right output size."""
    model = MLPSurrogate(n_inputs=4, n_outputs=64, hidden=(32, 32))
    x = torch.randn(8, 4)
    y = model(x)
    assert y.shape == (8, 64)

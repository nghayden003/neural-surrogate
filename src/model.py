"""
model.py
========

The neural surrogate model.

Architecture choice — why a plain MLP:
    The task is a fixed-size regression: a 4-vector of physical parameters maps
    to a fixed-length temperature profile. There is no sequence, no spatial
    translation-invariance to exploit on the input side, so the simplest
    sufficient model is a multilayer perceptron. Reaching for a CNN or
    transformer here would be over-engineering. (For a variable-length output,
    a full space-time field, or many coupled species, the natural next steps
    would be CNNs / Fourier Neural Operators - noted in the README as future work.)
"""

import torch
import torch.nn as nn


class MLPSurrogate(nn.Module):
    """
    A small fully-connected network mapping input parameters -> output field.

    Parameters
    ----------
    n_inputs : int
        Number of input parameters (4: D, amp, centre, width).
    n_outputs : int
        Length of the output temperature profile (= number of spatial points).
    hidden : tuple of int
        Sizes of the hidden layers.
    """

    def __init__(self, n_inputs=4, n_outputs=64, hidden=(128, 128)):
        super().__init__()
        layers = []
        prev = n_inputs
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())  # To account for the non-linearity, default lightweight, cost-effective option
            prev = h
        layers.append(nn.Linear(prev, n_outputs))  # linear output for regression (sigmoid for binary classification)
        self.net = nn.Sequential(*layers) # Save the network of layers in the model

    def forward(self, x):
        return self.net(x)

"""
train.py
========

Train the heat-diffusion neural surrogate.

Run:
    python -m src.train          # from the project root

What this script does, and why:
    1. Loads config (hyperparameters live in config.yaml, not in the code, so
       experiments are reproducible and changeable without editing source).
    2. Sets all random seeds (NumPy + PyTorch) for reproducibility.
    3. Generates data by simulation, then normalises inputs and outputs using
       TRAINING-SET statistics only (computing stats on the whole dataset would
       leak information from val/test into training).
    4. Trains an MLP with MSE loss (the natural choice for real-valued
       regression) and the Adam optimiser, recording train and validation loss
       each epoch so the curves can be inspected for under/over-fitting.
    5. Saves the trained weights, the normalisation statistics, and the loss
       curve to the outputs/ folder.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless backend so that we save figures, not show them
import matplotlib.pyplot as plt

from src.data import (
    generate_dataset, split_dataset, compute_stats, normalise,
)
from src.model import MLPSurrogate


def set_seeds(seed):
    """Seed NumPy and PyTorch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)

# Load configurations (such as seed, number of samples, batches and epochs)
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Convert array to PyTorch tensor for training
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32) # Float 32 is default in ML, efficient for GPU

# Run the training and validation process
def main():
    cfg = load_config()
    set_seeds(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    # Generate and split the data
    X, Y, x_grid = generate_dataset(
        n_samples=cfg["n_samples"],
        n_points=cfg["n_points"],
        t_final=cfg["t_final"],
        seed=cfg["seed"],
    )
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split_dataset(
        X, Y,
        frac_train=cfg["frac_train"],
        frac_val=cfg["frac_val"],
        seed=cfg["seed"],
    )

    # Normalisation statistics from the TRAINING set only. 
    # Do not extract normalisation from validation set to prevent validation data leakage. 
    x_mean, x_std = compute_stats(Xtr)
    y_mean, y_std = compute_stats(Ytr)

    Xtr_n = to_tensor(normalise(Xtr, x_mean, x_std))
    Ytr_n = to_tensor(normalise(Ytr, y_mean, y_std))
    Xva_n = to_tensor(normalise(Xva, x_mean, x_std))
    Yva_n = to_tensor(normalise(Yva, y_mean, y_std))

    # Create the surrogate model
    model = MLPSurrogate(
        n_inputs=X.shape[1],
        n_outputs=Y.shape[1],
        hidden=tuple(cfg["hidden"]),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])  # optimise with default Adam optimiser
    loss_fn = nn.MSELoss()  # Calculate loss

    # Train the model over all epochs
    n = Xtr_n.shape[0]
    batch = cfg["batch_size"]
    train_curve, val_curve = [], []

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_n[idx], Ytr_n[idx]

            optimiser.zero_grad() # Set initial zero gradient
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward() # Trace the gradient from the loss backwards
            optimiser.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n

        # validation loss (no gradient)
        model.eval()
        with torch.no_grad():  # Explicitly do not learn from gradient, it is validation not training
            val_loss = loss_fn(model(Xva_n), Yva_n).item()

        train_curve.append(epoch_loss)
        val_curve.append(val_loss)

        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"epoch {epoch+1:4d}/{cfg['epochs']}  "
                  f"train {epoch_loss:.4e}  val {val_loss:.4e}")

    # Save the model and normalisation statistics
    torch.save(model.state_dict(), "outputs/model.pt")
    np.savez(
        "outputs/norm_stats.npz",
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        x_grid=x_grid,
        Xte=Xte, Yte=Yte,           # stash the untouched test set for evaluate.py
    )

    # Plot the loss curve for inspection of whether hyperparameter finetuning is needed
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_curve, label="train")
    plt.semilogy(val_curve, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalised units)")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png", dpi=130)
    print("\nSaved model, normalisation stats and loss curve to outputs/.")


if __name__ == "__main__":
    main()

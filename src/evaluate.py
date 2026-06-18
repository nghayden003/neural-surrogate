"""
evaluate.py
===========

Evaluate the trained surrogate on the held-out test set.

Run (after train.py):
    python -m src.evaluate

Produces metrics that investigates the three aspects:
    1. Accuracy: how close are surrogate predictions to the true solver?
        (RMSE and mean relative error on the untouched test set)
    2. Qualitative check: do predicted profiles actually track the true
        diffused fields? (overlaid plots of test set samples)
    3. Speed: how much faster is one network forward pass than one PDE
        solve? This is the entire point of a surrogate, so it is the headline result.
"""

import time
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import simulate, normalise, denormalise
from src.model import MLPSurrogate


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # Load the model data and test set 
    stats = np.load("outputs/norm_stats.npz")
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]
    x_grid = stats["x_grid"]
    Xte, Yte = stats["Xte"], stats["Yte"]

    # Reconstruct the model with the test set input/output structure
    model = MLPSurrogate(
        n_inputs=Xte.shape[1],
        n_outputs=Yte.shape[1],
        hidden=tuple(cfg["hidden"]),
    )
    model.load_state_dict(torch.load("outputs/model.pt"))
    model.eval()

    # Test the model on the test set
    Xte_n = torch.tensor(normalise(Xte, x_mean, x_std), dtype=torch.float32)
    with torch.no_grad():
        pred_n = model(Xte_n).numpy()
    pred = denormalise(pred_n, y_mean, y_std)  # Convert back to physical units

    # Calculate metrics
    rmse = np.sqrt(np.mean((pred - Yte) ** 2))
    # Calculate relative error normalised by the peak of each true profile (robust to
    # the near-zero tails of a diffused Gaussian), 1e-8 to prevent division by zero
    peak = np.max(np.abs(Yte), axis=1, keepdims=True) + 1e-8
    mean_rel_err = np.mean(np.abs(pred - Yte) / peak)

    print(f"Test-set RMSE (physical units):      {rmse:.4e}")
    print(f"Test-set mean relative error:        {mean_rel_err*100:.2f}%")

    # Plot the results of ML predictions vs data on the test set
    k = min(4, Xte.shape[0])
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.2), sharey=True)
    if k == 1:
        axes = [axes]
    for j in range(k):
        axes[j].plot(x_grid, Yte[j], label="true solver", lw=2)
        axes[j].plot(x_grid, pred[j], "--", label="surrogate", lw=2)
        D, amp, centre, width = Xte[j]
        axes[j].set_title(f"D={D:.2f}, A={amp:.2f}\nc={centre:.2f}, w={width:.2f}",
                          fontsize=9)
        axes[j].set_xlabel("x")
        if j == 0:
            axes[j].set_ylabel("u(x, t_final)")
            axes[j].legend(fontsize=8)
    fig.suptitle("Surrogate vs. true diffused temperature field (test set)")
    fig.tight_layout()
    fig.savefig("outputs/predictions.png", dpi=130)

    # Benchmarking computational speed
    # True numerical solver: time one PDE solve per test input
    t0 = time.perf_counter()
    for p in Xte:
        simulate(p, x_grid, cfg["t_final"])
    t_solver = (time.perf_counter() - t0) / len(Xte)

    # Surrogate: time one forward pass per test input (batched=1 for fairness)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(Xte.shape[0]):
            model(Xte_n[i:i + 1])
    t_surrogate = (time.perf_counter() - t0) / Xte.shape[0]

    speedup = t_solver / t_surrogate
    print(f"\nMean time per true PDE solve:        {t_solver*1e3:.3f} ms")
    print(f"Mean time per surrogate forward pass:{t_surrogate*1e3:.3f} ms")
    print(f"Speed-up:                            {speedup:.0f}x")

    # Summarise the metrics in a separate text file
    with open("outputs/summary.txt", "w") as f:
        f.write(f"rmse={rmse:.6e}\n")
        f.write(f"mean_rel_err_pct={mean_rel_err*100:.3f}\n")
        f.write(f"t_solver_ms={t_solver*1e3:.4f}\n")
        f.write(f"t_surrogate_ms={t_surrogate*1e3:.4f}\n")
        f.write(f"speedup={speedup:.1f}\n")
    print("\nSaved predictions.png and summary.txt to outputs/.")


if __name__ == "__main__":
    main()

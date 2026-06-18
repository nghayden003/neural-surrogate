"""
analyse_errors.py
=================

Per-example error analysis for the heat-diffusion surrogate.

Purpose: the metrics after running src.evaluate is a mean over the test set. 
This script looks at the distribution of errors across individual examples, 
ranks the worst ones, and checks whether the worst predictions sit near the 
edges of the training parameter ranges (the expected failure mode for a data-driven surrogate), 
which interpolates well in the interior of its training distribution and less well at the edges.

Run (must be after src.train):
    python -m src.analyse_errors
"""

import yaml
import numpy as np
import torch

from src.data import normalise, denormalise
from src.model import MLPSurrogate


# The parameter ranges used in data.generate_dataset (D, amp, centre, width).
# Kept here so we can measure how close each sample sits to a range edge.
PARAM_RANGES = {
    "D":      (0.1, 1.0),
    "amp":    (0.5, 2.0),
    "centre": (3.0, 7.0),
    "width":  (0.5, 1.5),
}
PARAM_NAMES = ["D", "amp", "centre", "width"]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def edge_proximity(params):
    """
    How close a parameter set sits to the edge of its sampled box.

    For each parameter, map its value to [0, 1] within its range, then measure
    distance to the nearest edge: 0.5 means dead-centre, 0.0 means hard on an
    edge. We return the minimum over the four parameters - i.e. how close the
    single most extreme parameter is to its boundary. 
    """
    closeness = []
    for name, value in zip(PARAM_NAMES, params):
        lo, hi = PARAM_RANGES[name]
        frac = (value - lo) / (hi - lo)          # position in range, 0..1
        closeness.append(min(frac, 1.0 - frac))  # distance to nearest edge
    return min(closeness)


def main():
    cfg = load_config()

    # Load the statistics and test set (similar to evaluate.py)
    stats = np.load("outputs/norm_stats.npz")
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]
    Xte, Yte = stats["Xte"], stats["Yte"]

    model = MLPSurrogate(
        n_inputs=Xte.shape[1],
        n_outputs=Yte.shape[1],
        hidden=tuple(cfg["hidden"]),
    )
    model.load_state_dict(torch.load("outputs/model.pt"))
    model.eval()

    # Predict results on the test set
    Xte_n = torch.tensor(normalise(Xte, x_mean, x_std), dtype=torch.float32)
    with torch.no_grad():
        pred = denormalise(model(Xte_n).numpy(), y_mean, y_std)

    # Calculate per-example relative error 
    # Same definition as evaluate.py, but kept PER EXAMPLE rather than averaged:
    # error of each profile, relative to that profile's own peak.
    peak = np.max(np.abs(Yte), axis=1) + 1e-8  # (N,)
    per_example_err = np.mean(np.abs(pred - Yte), axis=1) / peak  # (N,)

    # Summarise the error distribution
    print("Per-example relative error (%) distribution over the test set:")
    print(f"  mean   : {per_example_err.mean()*100:.3f}")
    print(f"  median : {np.median(per_example_err)*100:.3f}")
    print(f"  min    : {per_example_err.min()*100:.3f}")
    print(f"  max    : {per_example_err.max()*100:.3f}")
    print(f"  90th pct: {np.percentile(per_example_err, 90)*100:.3f}")

    # Print the five worse-predicted test samples with their parameters
    order = np.argsort(per_example_err)[::-1]   # worst first
    print("\n5 worst-predicted test examples:")
    print(f"{'rel_err%':>9} | {'D':>5} {'amp':>5} {'centre':>6} {'width':>5} | edge-dist")
    for i in order[:5]:
        D, amp, centre, width = Xte[i]
        print(f"{per_example_err[i]*100:9.3f} | {D:5.2f} {amp:5.2f} {centre:6.2f} "
              f"{width:5.2f} | {edge_proximity(Xte[i]):.3f}")

    # Print the five best-predicted test samples for comparison
    print("\n5 best-predicted test examples:")
    print(f"{'rel_err%':>9} | {'D':>5} {'amp':>5} {'centre':>6} {'width':>5} | edge-dist")
    for i in order[::-1][:5]:
        D, amp, centre, width = Xte[i]
        print(f"{per_example_err[i]*100:9.3f} | {D:5.2f} {amp:5.2f} {centre:6.2f} "
              f"{width:5.2f} | {edge_proximity(Xte[i]):.3f}")

    # Check if error correlates with edge proximity
    # Negative correlation between error and edge-distance -> 
    # the closer to an edge (smaller edge-dist), the larger the error.
    edge_dists = np.array([edge_proximity(p) for p in Xte])
    corr = np.corrcoef(per_example_err, edge_dists)[0, 1]
    print(f"\nCorrelation(rel_error, edge-distance) = {corr:+.3f}")
    print("  (negative => errors grow as samples approach the range edges)")

    # Another check: is the narrow-width profiles specifically harder to predict? 
    # Split at the median width and compare mean error in each half.
    med_w = np.median(Xte[:, 3])
    narrow = Xte[:, 3] < med_w
    print(f"\nMean rel-error, narrow-width half (w < {med_w:.2f}): "
          f"{per_example_err[narrow].mean()*100:.3f}%")
    print(f"Mean rel-error, wide-width half  (w >= {med_w:.2f}): "
          f"{per_example_err[~narrow].mean()*100:.3f}%")


if __name__ == "__main__":
    main()

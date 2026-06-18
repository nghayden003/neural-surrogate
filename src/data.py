"""
data.py
=======

Data generation for the heat-diffusion neural surrogate.

The "true" simulator solves the 1-D heat equation

    u_t = D * u_xx

on a rod of fixed length, with fixed-temperature (Dirichlet, u = 0) ends, starting
from a Gaussian initial temperature profile. The surrogate will learn the map

    (D, amplitude, centre, width)  ->  u(x) at a fixed final time t_final

i.e. physical parameters + initial condition -> diffused temperature field.

This is the pure-diffusion skeleton of a cosmic-ray transport equation (without the
advection, energy-loss and source terms), so it is a deliberately minimal analogue 
of a full particle transport solver.

Why solve with an implicit method (BDF):
    Explicit schemes for diffusion are stability-limited by the CFL condition
    (the time step must shrink as dx^2). BDF is an implicit integrator which is stable from stiffness, 
    so the solve stays robust for every diffusion coefficient we sample without hand-tuning the step size. 
"""

import numpy as np
from scipy.integrate import solve_ivp


def simulate(params, x_grid, t_final):
    """
    Solve the 1-D heat equation from t = 0 to ``t_final`` for one parameter set.

    Parameters
    ----------
    params : sequence of float
        [D, amp, centre, width]:
        D      - diffusion coefficient
        amp    - peak height of the Gaussian initial profile
        centre - position of the initial peak along the rod
        width  - standard deviation of the initial Gaussian
    x_grid : np.ndarray
        Fixed spatial grid (defines the length of the rod).
    t_final : float
        Time at which the temperature profile is returned.

    Returns
    -------
    np.ndarray
        Temperature profile u(x) at ``t_final``, same length as ``x_grid``.
    """
    D, amp, centre, width = params
    dx = x_grid[1] - x_grid[0]

    # Initial condition: a Gaussian temperature profile
    u0 = amp * np.exp(-((x_grid - centre) ** 2) / (2.0 * width ** 2))

    # Method of lines: discretise space (central differences for d2u/dx2),
    # then integrate the resulting ODE system in time. Interior points only ([1:-1]);
    # the boundaries are held at 0 (Dirichlet), so their derivative stays 0.
    def rhs(t, u):
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx ** 2 # Central-difference approximation
        return D * d2u

    sol = solve_ivp(
        rhs,
        (0.0, t_final),
        y0=u0,
        t_eval=[t_final],
        method="BDF",  # implicit, stable for stiff diffusion
    )
    return sol.y[:, -1]


# Sample the parameter space and generate simulated data for training, validation and testing
def generate_dataset(n_samples, n_points=64, t_final=2.0, seed=0):
    """
    Generate (X, Y) training data by sampling parameters and simulating.

    Parameters
    ----------
    n_samples : int
        Number of (input, output) pairs to generate.
    n_points : int
        Number of spatial grid points = length of each output vector.
    t_final : float
        Final time at which the temperature field is recorded.
    seed : int
        RNG seed for reproducible sampling.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
        Input parameters [D, amp, centre, width].
    Y : np.ndarray, shape (n_samples, n_points)
        Output temperature profiles at ``t_final``.
    x_grid : np.ndarray, shape (n_points,)
        The spatial grid (useful for plotting).
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(0.0, 10.0, n_points)

    # Sample physical parameters uniformly over sensible ranges.
    D = rng.uniform(0.1, 1.0, n_samples)        # diffusion coefficient
    amp = rng.uniform(0.5, 2.0, n_samples)      # initial peak height
    centre = rng.uniform(3.0, 7.0, n_samples)   # initial peak position
    width = rng.uniform(0.5, 1.5, n_samples)    # initial peak width

    X = np.stack([D, amp, centre, width], axis=1)            # (N, 4)
    Y = np.array([simulate(p, x_grid, t_final) for p in X])  # (N, n_points)
    return X, Y, x_grid


# Normalisation helper functions:
# Standardise (mean 0, std 1) so features on different scales contribute comparably, 
# giving balanced gradients and faster, more stable convergence.

def compute_stats(arr):
    """Return per-column mean and std for standardisation.

    A small epsilon guards against division by zero on constant columns.
    """
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return mean, std

def normalise(arr, mean, std):
    """Standardise: (arr - mean) / std."""
    return (arr - mean) / std


def denormalise(arr, mean, std):
    """Invert standardisation: arr * std + mean (predictions -> physical units)."""
    return arr * std + mean


# Split the dataset into training (70%), validation (15%), and test (15%).
def split_dataset(X, Y, frac_train=0.7, frac_val=0.15, seed=0):
    """
    Shuffle and split into train / validation / test subsets.

    Why three sets:
        - train: fit the model weights
        - validation:   tune choices and watch for overfitting during development
        - test:  touched once, at the end, for an honest generalisation score
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)

    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]

    return (X[tr], Y[tr]), (X[va], Y[va]), (X[te], Y[te])

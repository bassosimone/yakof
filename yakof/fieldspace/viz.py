"""
Field Space Visualization
========================

This module provides visualization tools for analyzing field distributions
and their statistics. It builds on matplotlib to visualize field tensors
that combine time and ensemble dimensions.

The module supports:
1. Distribution plots showing quantiles over time
2. Confidence interval plots
3. Individual ensemble member path plots

Example:
    >>> model = Model()
    >>> field = model.field.normal_rvs(shape=(100, 24), loc=0.0, scale=1.0)
    >>> ctx = backend.numpy_engine.PartialEvaluationContext()
    >>> plot_distribution(field, model, ctx)

The visualization functions expect field tensors created through the orientation
system and require an evaluation context for computing actual values.

SPDX-License-Identifier: Apache-2.0
"""

from typing import Optional, Sequence

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes

from ..backend import numpy_engine, oriented
from .model import Model, FieldWise


def plot_distribution(
    field: oriented.Tensor[FieldWise],
    model: Model,
    ctx: numpy_engine.PartialEvaluationContext,
    ax: Axes,  # Make ax required
    title: str,  # Make title required
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    cmap: str = "Blues",
    alpha: float = 0.3,
    **kwargs,
) -> None:  # Return None like phasespace
    """Plot distribution of field values over time.

    Args:
        field: Field tensor to visualize
        model: Model containing timestamps
        ctx: Evaluation context with bindings
        ax: Matplotlib axis to plot on
        title: Plot title
        quantiles: Quantiles to show
        cmap: Colormap for quantile fills
        alpha: Transparency for fills
        **kwargs: Additional kwargs passed to plot

    Raises:
        TypeError: If inputs have incorrect types
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib.axes.Axes, got {type(ax)}")
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title)}")

    # Evaluate field tensor
    data = ctx.evaluate(field.t)

    # Calculate quantiles
    q_values = np.quantile(data, quantiles, axis=0)

    # Plot median line
    median_idx = len(quantiles) // 2
    ax.plot(q_values[median_idx], **kwargs)

    # Plot quantile bands
    for i in range(len(quantiles) // 2):
        ax.fill_between(
            range(data.shape[1]),
            q_values[i],
            q_values[-(i + 1)],
            alpha=alpha,
            color=plt.get_cmap(cmap)(i / (len(quantiles) // 2)),
        )

    ax.set_title(title)


def plot_confidence_intervals(
    field: oriented.Tensor[FieldWise],
    model: Model,
    ctx: numpy_engine.PartialEvaluationContext,
    ax: Axes,  # Make ax required
    title: str,  # Make title required
    confidence_level: float = 0.95,
    **kwargs,
) -> None:
    """Plot confidence intervals for field over time.

    Args:
        field: Field tensor to visualize
        model: Model containing timestamps
        ctx: Evaluation context with bindings
        ax: Matplotlib axis to plot on
        title: Plot title
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        **kwargs: Additional kwargs for plot

    Raises:
        TypeError: If inputs have incorrect types
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib.axes.Axes, got {type(ax)}")
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title)}")

    # Evaluate field tensor
    data = ctx.evaluate(field.t)

    # Calculate confidence bounds
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower = mean - z_score * std
    upper = mean + z_score * std

    # Plot
    ax.plot(mean, **kwargs)
    ax.fill_between(range(data.shape[1]), lower, upper, alpha=0.3)
    ax.set_title(title)


def plot_ensemble_paths(
    field: oriented.Tensor[FieldWise],
    model: Model,
    ctx: numpy_engine.PartialEvaluationContext,
    ax: Axes,  # Make ax required
    title: str,  # Make title required
    n_paths: int = 10,
    alpha: float = 0.3,
    **kwargs,
) -> None:
    """Plot individual paths from the ensemble.

    Args:
        field: Field tensor to visualize
        model: Model containing timestamps
        ctx: Evaluation context with bindings
        ax: Matplotlib axis to plot on
        title: Plot title
        n_paths: Number of random paths to show
        alpha: Transparency for individual paths
        **kwargs: Additional kwargs for plot

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If n_paths is less than 1
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib.axes.Axes, got {type(ax)}")
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title)}")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    # Evaluate field tensor
    data = ctx.evaluate(field.t)

    # Randomly select paths
    indices = np.random.choice(data.shape[0], size=n_paths, replace=False)

    # Plot paths
    for idx in indices:
        ax.plot(data[idx], alpha=alpha, **kwargs)

    ax.set_title(title)

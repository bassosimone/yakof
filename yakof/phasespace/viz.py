"""
Phase Space Visualization
========================

Tools for visualizing phase space analysis results.

The main visualization shows:
1. The mean value of observables across parameter space (color map)
2. The uncertainty in these values (contour lines)

Note: We make NO assumptions about the distribution of the output values.
The uncertainty contours show standard deviation, but this should not be
interpreted as implying any specific probability distribution.

SPDX-License-Identifier: Apache-2.0
"""

from typing import TypeAlias
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np

from .analysis import Result

# Type aliases for clarity
ColorMap: TypeAlias = str
ExtentType: TypeAlias = tuple[float, float, float, float]


def plot(
    result: Result,
    observable: str,
    ax: Axes,  # Make ax required
    title: str | None = None,
    cmap: ColorMap = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> None:
    """Plot 2D phase diagram.

    Args:
        result: Phase space analysis result
        observable: Name of observable to plot (must be a 2D array)
        ax: Matplotlib axis to plot on
        title: Optional title for the plot
        cmap: Colormap to use
        vmin, vmax: Optional value range for colormap
        **kwargs: Additional kwargs for imshow

    Raises:
        ValueError: If observable doesn't exist or has wrong dimensions
        TypeError: If inputs have incorrect types
    """
    if not isinstance(observable, str):
        raise TypeError(f"observable must be str, got {type(observable)}")
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib.axes.Axes, got {type(ax)}")

    if observable not in result.observables:
        raise ValueError(f"Observable '{observable}' not found in result")

    # Get parameter names
    param_names = list(result.parameters.keys())
    if len(param_names) != 2:
        raise ValueError("plot requires exactly 2 parameters")

    # Get data and validate shape
    data = result.observables[observable]
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Observable data must be ndarray, got {type(data)}")
    if len(data.shape) != 2:
        raise ValueError(
            f"Observable '{observable}' must be 2D, got shape {data.shape}"
        )

    extent = (
        result.parameters[param_names[0]][0],
        result.parameters[param_names[0]][-1],
        result.parameters[param_names[1]][0],
        result.parameters[param_names[1]][-1],
    )

    # Create plot
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    plt.colorbar(im, ax=ax)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    if title:
        ax.set_title(title)


def plot_with_contours(
    result: Result,
    observable: str,
    ax: Axes,
    title: str,
    cmap: ColorMap = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> None:
    """Plot uncertainty contours.

    Args:
        result: Analysis result containing mean and uncertainty data
        observable: Name of the observable to plot
        ax: Matplotlib axis to plot on
        title: Plot title
        cmap: Colormap for mean values
        vmin, vmax: Optional colormap value range

    Note:
        The contour lines show standard deviation but this does NOT imply
        any specific probability distribution of the observable values.
        Standard deviation is used as a general measure of uncertainty.

    Raises:
        ValueError: If observable doesn't exist or has wrong dimensions
        TypeError: If inputs have incorrect types
    """
    if not isinstance(observable, str):
        raise TypeError(f"observable must be str, got {type(observable)}")
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib.axes.Axes, got {type(ax)}")

    mean = result.observables[observable]
    if not isinstance(mean, np.ndarray):
        raise TypeError(f"Observable data must be ndarray, got {type(mean)}")

    std = result.std(observable)
    if std is not None and not isinstance(std, np.ndarray):
        raise TypeError(f"Std dev must be ndarray or None, got {type(std)}")

    # Get parameter names
    param_names = list(result.parameters.keys())

    # Plot mean as background color
    im = ax.imshow(
        mean,
        origin="lower",
        extent=(0, 100, 0, 100),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    if std is not None:
        x = np.linspace(0, 100, mean.shape[1])
        y = np.linspace(0, 100, mean.shape[0])

        # Add contour lines for uncertainty (standard deviation)
        std_levels = np.linspace(std.min(), std.max(), 5)
        CS = ax.contour(
            x, y, std, colors="w", alpha=0.7, levels=std_levels, linewidths=0.5
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="std=%.2f")

    plt.colorbar(im, ax=ax)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.minorticks_on()
    ax.set_title(title)

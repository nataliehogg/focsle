"""
Plotting utilities for Fisher forecast results.

This module provides functions for visualizing Fisher matrix constraints
and comparing results from different datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def plot_fisher_ellipse(
    F: np.ndarray,
    fiducial: Tuple[float, float],
    ax: plt.Axes,
    color: str = 'steelblue',
    label: Optional[str] = None,
    linestyle: str = '-',
    alpha_1sig: float = 0.3,
    alpha_2sig: float = 0.15,
    show_2sigma: bool = True,
) -> bool:
    """
    Plot 1-sigma and 2-sigma ellipses for a Fisher matrix.

    Args:
        F: 2x2 Fisher matrix
        fiducial: (x, y) fiducial parameter values
        ax: Matplotlib axes to plot on
        color: Color for the ellipse
        label: Label for legend
        linestyle: Line style for ellipse edge
        alpha_1sig: Alpha for 1-sigma fill
        alpha_2sig: Alpha for 2-sigma fill
        show_2sigma: Whether to show 2-sigma contour

    Returns:
        True if successful, False if Fisher matrix is singular
    """
    try:
        C = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        print(f"Warning: Singular Fisher matrix for {label}")
        return False

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Ensure positive eigenvalues (numerical precision issues)
    if np.any(eigenvalues < 0):
        print(f"Warning: Negative eigenvalues for {label}, using absolute values")
        eigenvalues = np.abs(eigenvalues)

    # Ellipse angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Plot contours
    sigmas = [1, 2] if show_2sigma else [1]
    alphas = [alpha_1sig, alpha_2sig]
    linestyles = [linestyle, '-']

    for i, n_sigma in enumerate(sigmas):
        width = 2 * n_sigma * np.sqrt(eigenvalues[0])
        height = 2 * n_sigma * np.sqrt(eigenvalues[1])

        ellipse = Ellipse(
            fiducial, width, height,
            angle=np.degrees(angle),
            facecolor=color,
            alpha=alphas[i],
            edgecolor=color,
            linewidth=2,
            linestyle=linestyles[i],
            label=label if n_sigma == 1 else None
        )
        ax.add_patch(ellipse)

    return True


def plot_constraints(
    results: Dict,
    probes: List[str] = None,
    figsize: Tuple[float, float] = None,
    colors: Dict[str, str] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Fisher ellipse constraints for multiple probes.

    Args:
        results: Dictionary containing Fisher matrices and fiducial values
        probes: List of probes to plot (default: ['LL', 'LE', 'LP', 'Combined'])
        figsize: Figure size (width, height)
        colors: Dictionary mapping probe names to colors
        output_file: If provided, save figure to this path
        title: Optional figure title

    Returns:
        Matplotlib figure
    """
    if probes is None:
        probes = ['LL', 'LE', 'LP', 'Combined']

    # Filter to available probes
    probes = [p for p in probes if p in results.get('fisher_matrices', {})]

    if not probes:
        raise ValueError("No valid probes found in results")

    # Default colors
    if colors is None:
        colors = {
            'LL': 'steelblue',
            'LE': 'forestgreen',
            'LP': 'coral',
            'Combined': 'purple',
        }

    # Set up figure
    n_plots = len(probes)
    if figsize is None:
        figsize = (6 * n_plots, 5)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    fiducial = results['fiducial']

    probe_titles = {
        'LL': 'LOS--LOS',
        'LE': 'LOS--shape',
        'LP': 'LOS--position',
        'Combined': 'Combined',
    }

    for ax, probe in zip(axes, probes):
        F = results['fisher_matrices'][probe]
        color = colors.get(probe, 'steelblue')

        plot_fisher_ellipse(
            F, fiducial, ax,
            color=color,
            label=probe,
        )

        # Mark fiducial
        ax.plot(fiducial[0], fiducial[1], 'k+', markersize=10, markeredgewidth=2)

        # Labels
        ax.set_xlabel(r'$\Omega_m$', fontsize=14)
        ax.set_ylabel(r'$\sigma_8$', fontsize=14)
        ax.set_title(f'{probe}: {probe_titles.get(probe, probe)}', fontsize=13)
        ax.legend(loc='best', fontsize=11)
        ax.grid(False)

        # Auto-scale axes
        ax.autoscale()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    return fig


def plot_comparison(
    results_list: List[Dict],
    labels: List[str],
    probes: List[str] = None,
    figsize: Tuple[float, float] = None,
    colors: List[str] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Compare Fisher constraints from multiple datasets.

    Args:
        results_list: List of results dictionaries
        labels: Labels for each dataset
        probes: List of probes to plot (default: ['LL', 'LE', 'LP'])
        figsize: Figure size
        colors: List of colors for each dataset
        output_file: If provided, save figure to this path
        title: Optional figure title

    Returns:
        Matplotlib figure
    """
    if probes is None:
        probes = ['LL', 'LE', 'LP']

    if colors is None:
        colors = ['steelblue', 'crimson', 'forestgreen', 'purple', 'orange']

    n_plots = len(probes)
    if figsize is None:
        figsize = (6 * n_plots, 5)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    fiducial = results_list[0]['fiducial']

    probe_titles = {
        'LL': 'LOS--LOS',
        'LE': 'LOS--shape',
        'LP': 'LOS--position',
        'Combined': 'Combined',
    }

    for ax, probe in zip(axes, probes):
        for i, (results, label) in enumerate(zip(results_list, labels)):
            if probe not in results.get('fisher_matrices', {}):
                continue

            F = results['fisher_matrices'][probe]
            color = colors[i % len(colors)]

            plot_fisher_ellipse(
                F, fiducial, ax,
                color=color,
                label=label,
                alpha_1sig=0.3 - 0.05 * i,
                alpha_2sig=0.15 - 0.03 * i,
            )

        # Mark fiducial
        ax.plot(fiducial[0], fiducial[1], 'k+', markersize=10, markeredgewidth=2)

        ax.set_xlabel(r'$\Omega_m$', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel(r'$\sigma_8$', fontsize=14)
        ax.set_title(f'{probe}: {probe_titles.get(probe, probe)}', fontsize=13)
        ax.grid(False)
        ax.autoscale()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # Single legend outside axes (collect unique handles/labels from all axes)
    handles = []
    legend_labels = []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=max(1, len(legend_labels)),
                   frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_file}")

    return fig


def print_constraints_table(results: Dict, probes: List[str] = None):
    """
    Print a formatted table of parameter constraints.

    Args:
        results: Dictionary containing constraints
        probes: List of probes to include (default: all available)
    """
    if probes is None:
        probes = list(results.get('constraints', {}).keys())

    fiducial = results['fiducial']

    print("\n" + "=" * 70)
    print("Parameter Constraints Summary")
    print("=" * 70)
    print(f"Fiducial: Omega_m = {fiducial[0]:.4f}, sigma_8 = {fiducial[1]:.4f}")
    print("-" * 70)
    print(f"{'Probe':<12} {'sigma(Om)':<12} {'%(Om)':<10} {'sigma(s8)':<12} {'%(s8)':<10} {'corr':<8}")
    print("-" * 70)

    for probe in probes:
        c = results['constraints'].get(probe)
        if c is None:
            print(f"{probe:<12} {'SINGULAR':<12}")
            continue

        print(f"{probe:<12} "
              f"{c['errors'][0]:<12.4f} "
              f"{100 * c['fractional_errors'][0]:<10.1f} "
              f"{c['errors'][1]:<12.4f} "
              f"{100 * c['fractional_errors'][1]:<10.1f} "
              f"{c['correlation']:<8.3f}")

    print("=" * 70)


def plot_fom_comparison(
    results_list: List[Dict],
    labels: List[str],
    probes: List[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_file: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Figure of Merit comparison as a bar chart.

    Args:
        results_list: List of results dictionaries
        labels: Labels for each dataset
        probes: List of probes to compare
        figsize: Figure size
        output_file: If provided, save figure to this path

    Returns:
        Matplotlib figure
    """
    if probes is None:
        probes = ['LL', 'LE', 'LP', 'Combined']

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(probes))
    width = 0.8 / len(results_list)

    for i, (results, label) in enumerate(zip(results_list, labels)):
        foms = []
        for probe in probes:
            F = results['fisher_matrices'].get(probe)
            if F is not None:
                fom = np.sqrt(np.linalg.det(F))
                foms.append(fom)
            else:
                foms.append(0)

        offset = (i - len(results_list) / 2 + 0.5) * width
        bars = ax.bar(x + offset, foms, width, label=label)

    ax.set_xlabel('Probe', fontsize=12)
    ax.set_ylabel('Figure of Merit', fontsize=12)
    ax.set_title('Figure of Merit Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(probes)
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved FoM comparison to {output_file}")

    return fig

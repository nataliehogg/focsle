#!/usr/bin/env python
"""Plot optimistic and conservative Euclid LLLELP w0-wa forecasts."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import (
    marginalize_fisher_to_pair,
    plot_fisher_ellipse,
)
from sanglier.palettes import cool, warm


try:
    plt.style.use('sanglier')
except OSError:
    pass


from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', size=11)


PARAMETER_PAIR = ('w0', 'wa')
FISHER_KEY = 'LL+LE+LP'
OUTPUT_FILE = PROJECT_ROOT / 'figures' / 'euclid_LLLELP_w0wa_contours.pdf'

SCENARIOS = (
    {
        'title': r'(a) Optimistic',
        'color': warm[2],
        'results_file': (
            ROOT / 'results' /
            'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse_LLLELP_w0wa.pkl'
        ),
    },
    {
        'title': r'(b) Conservative',
        'color': cool[1],
        'results_file': (
            ROOT / 'results' /
            'EUCLID-ENV-CON_fine_qmc1pct_scaled_inverse_LLLELP_w0wa.pkl'
        ),
    },
)


def load_marginalized_forecast(config):
    """Load one four-parameter result and marginalize it to w0-wa."""
    results = FisherForecast.load_results(config['results_file'])
    if FISHER_KEY not in results['fisher_matrices']:
        raise KeyError(
            f'{FISHER_KEY} is missing from {config["results_file"]}'
        )

    fisher, fiducial = marginalize_fisher_to_pair(
        results['fisher_matrices'][FISHER_KEY],
        results['fiducial'],
        results['param_names'],
        PARAMETER_PAIR,
    )
    errors = dict(zip(
        results['param_names'],
        results['constraints'][FISHER_KEY]['errors'],
    ))
    return fisher, fiducial, errors


def axis_limits(fisher, fiducial, margin=2.8):
    """Return limits containing the joint 95 per cent contour."""
    covariance = np.linalg.inv(fisher)
    errors = np.sqrt(np.diag(covariance))
    return (
        (fiducial[0] - margin * errors[0],
         fiducial[0] + margin * errors[0]),
        (fiducial[1] - margin * errors[1],
         fiducial[1] + margin * errors[1]),
    )


fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))

for ax, config in zip(axes, SCENARIOS):
    fisher, fiducial, errors = load_marginalized_forecast(config)

    plot_fisher_ellipse(
        fisher,
        fiducial,
        ax,
        color=config['color'],
        label=r'Euclid DR3 LLLELP',
        show_2sigma=True,
        zorder=20,
    )
    ax.plot(
        *fiducial,
        marker='+',
        color='black',
        markersize=7,
        markeredgewidth=1.3,
        zorder=30,
    )

    xlim, ylim = axis_limits(fisher, fiducial)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(config['title'])
    ax.set_xlabel(r'$w_0$')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_box_aspect(1)
    ax.legend(
        handles=[
            mpatches.Patch(
                color=config['color'],
                alpha=0.5,
                label=r'Euclid DR3 LLLELP',
            )
        ],
        loc='best',
        fontsize=9,
        frameon=False,
    )

    print(
        f"{config['title']}: "
        f"sigma(w0)={errors['w0']:.6g}, sigma(wa)={errors['wa']:.6g}"
    )

axes[0].set_ylabel(r'$w_a$')

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')

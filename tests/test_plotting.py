"""Tests for Fisher ellipse plotting conventions."""

import math

import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_usetex():
    """focsle.plotting enables usetex globally; disable for headless tests."""
    import focsle.plotting  # noqa: F401  (triggers the rc() calls)
    matplotlib.rcParams['text.usetex'] = False
    yield
    plt.close('all')


def test_fisher_ellipse_2d_confidence_scaling():
    """The drawn ellipses must be joint 2D 68.3%/95.4% regions.

    For F = C = identity, sqrt(eigenvalues) = 1, so the semi-axes must be
    exactly the chi^2(2 dof) scale factors: sqrt(-2 ln(1-p)) with
    p = erf(n/sqrt(2)) -> 1.5152 (1 sigma) and 2.4860 (2 sigma).
    Regression test for the n_sigma * sqrt(eig) bug (39%/86% content).
    """
    from focsle.plotting import plot_fisher_ellipse

    fig, ax = plt.subplots()
    ok = plot_fisher_ellipse(np.eye(2), (0.0, 0.0), ax, show_2sigma=True)
    assert ok

    ellipses = ax.patches
    assert len(ellipses) == 2

    expected = [
        math.sqrt(-2.0 * math.log(1.0 - math.erf(n / math.sqrt(2.0))))
        for n in (1, 2)
    ]
    np.testing.assert_allclose(expected, [1.5151, 2.4860], atol=2e-4)  # sanity

    for ellipse, semi_axis in zip(ellipses, expected):
        np.testing.assert_allclose(ellipse.width / 2, semi_axis, rtol=1e-12)
        np.testing.assert_allclose(ellipse.height / 2, semi_axis, rtol=1e-12)


def test_fisher_ellipse_anisotropic_covariance():
    """Semi-axes must scale with sqrt of the covariance eigenvalues."""
    from focsle.plotting import plot_fisher_ellipse

    # Diagonal covariance with variances 4 and 0.25 -> C = F^{-1}
    C = np.diag([4.0, 0.25])
    F = np.linalg.inv(C)

    fig, ax = plt.subplots()
    assert plot_fisher_ellipse(F, (0.0, 0.0), ax, show_2sigma=False)

    (ellipse,) = ax.patches
    scale_1sig = math.sqrt(-2.0 * math.log(1.0 - math.erf(1 / math.sqrt(2.0))))

    # eigh returns ascending eigenvalues: [0.25, 4.0]
    np.testing.assert_allclose(ellipse.width / 2, scale_1sig * 0.5, rtol=1e-12)
    np.testing.assert_allclose(ellipse.height / 2, scale_1sig * 2.0, rtol=1e-12)


def test_fisher_ellipse_linestyle_applies_to_both_contours():
    """The 2-sigma contour must honour the linestyle argument (was hardcoded '-')."""
    from focsle.plotting import plot_fisher_ellipse

    fig, ax = plt.subplots()
    assert plot_fisher_ellipse(np.eye(2), (0.0, 0.0), ax,
                               linestyle='--', show_2sigma=True)

    styles = {e.get_linestyle() for e in ax.patches}
    assert len(styles) == 1  # both contours share the requested style


def test_chainconsumer_cross_check():
    """Smoke test: the ChainConsumer wrapper builds a figure from Fisher results."""
    pytest.importorskip('chainconsumer')
    from focsle.plotting import plot_constraints_chainconsumer

    F = np.array([[4000.0, -1000.0], [-1000.0, 2000.0]])
    results = {
        'fisher_matrices': {'LL': F, 'LE': 2 * F},
        'fiducial': [0.31, 0.81],
    }

    fig = plot_constraints_chainconsumer(results, probes=['LL', 'LE'])
    assert fig is not None
    assert len(fig.axes) > 0

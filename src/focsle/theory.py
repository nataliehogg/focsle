"""
JAX-accelerated theory calculator for Fisher forecasts.

Everything except CAMB power spectrum calls is JAX-ified and runs on GPU.
Uses automatic differentiation for Fisher matrix derivatives.
"""

import numpy as np
import pickle
from pathlib import Path
import warnings
from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

import camb
from camb import model

# Try to use GPU if available, otherwise fall back to CPU cleanly
try:
    jax.config.update('jax_platform_name', 'gpu')
    print(f"JAX running on: {jax.devices()}")
except RuntimeError:
    print("GPU not available, falling back to CPU")
    jax.config.update('jax_platform_name', 'cpu')
    print(f"JAX running on: {jax.devices()}")


# =============================================================================
# Physical Constants and Default Grid Parameters
# =============================================================================

# Redshift grid defaults for P(k) computation
# Covers z(chi) over the full Limber integration range (chi_max = 8000 Mpc
# corresponds to z ~ 4.3); queries beyond the grid clamp to the edge.
Z_MAX_CAMB = 4.6  # Maximum redshift for CAMB P(k) grid
Z_MIN_CAMB = 0.0  # Minimum redshift (today)

# Wavenumber grid defaults for P(k) computation.
# NOTE units: plain 1/Mpc (the interpolator is built with hubble_units=False,
# k_hunit=False, and CAMB's set_matter_power kmax is also "just k, not k/h").
# Queries beyond K_MAX are power-law extrapolated in Pk_interp.
K_MIN_CAMB = 1e-4  # Minimum k (1/Mpc) - large scales
K_MAX_CAMB = 10.0  # Maximum k (1/Mpc) - small scales

# Comoving distance grid defaults (in Mpc)
# Range covers typical lens-source separations in strong lensing
CHI_MIN_DEFAULT = 10.0    # Minimum comoving distance (Mpc) - nearby universe
CHI_MAX_DEFAULT = 8000.0  # Maximum comoving distance (Mpc) - covers z~2-3

# Extended range for pre-computing mean Q_L kernel
CHI_MAX_QL_GRID = 10000.0  # Larger range for Q_L kernel grid

# Default number of grid points for numerical integrations
# N_CHI_CL: 400 keeps the sharp tomographic-bin edges in Q_P well resolved as
# they sweep through the chi grid with Omega_m; at 100 points the resulting
# ripples in D(Omega_m) made dD/dOm noisy at the tens-of-percent level for LP.
N_CHI_CL = 400   # Points for C_ell integrals over comoving distance
N_CHI_QL = 200   # Points for Q_L kernel pre-computation (higher accuracy needed)
N_Z_KERNEL = 256  # Points for redshift-bin averaging in Q_E/Q_P precomputation

# Angular multipole defaults: grid on which C_ell is computed and then
# log-log interpolated (power-law tails) inside the Ogata Hankel transforms.
# The wide range keeps the Ogata nodes ell = x_k/theta inside the grid for
# the data's angular range; beyond it the power-law tail takes over.
ELL_MIN_DEFAULT = 0.1      # Minimum multipole
ELL_MAX_DEFAULT = 1e6      # Maximum multipole
N_ELL_DEFAULT = 300        # Number of ell points for the C_ell grid

# Ogata (2005) quadrature settings for the Hankel transforms
# (same scheme and parameters as the `hankel` package used by loscov)
OGATA_N = 10000  # number of Bessel-zero nodes
OGATA_H = 1e-2   # step parameter


class TheoryJAX:
    """
    JAX-accelerated theory calculator.

    Pre-computes CAMB P(k) grid, then uses JAX for everything else.

    Args:
        lens_file: Path to lens catalog file. If None, uses fallback sample.
        cosmo_fid: Dictionary of fiducial cosmology parameters. If None, uses defaults.

    Example:
        >>> from focsle.theory import TheoryJAX
        >>> theory = TheoryJAX(lens_file='data/Euclid_lenses.txt')
        >>> theory.setup_Pk_grid(nOm=5, nAs=5)
        >>> theory.setup_galaxy_distributions('/path/to/data/dataset_dir')
    """

    def __init__(self, lens_file: Optional[str] = None,
                 cosmo_fid: Optional[Dict] = None):
        self.c_km_s = 299792.458  # km/s

        # Fiducial cosmology (can be overridden)
        if cosmo_fid is None:
            self.cosmo_fid = {
                'H0': 67.37,
                'ombh2': 0.0223,
                'omch2': 0.1198,
                'mnu': 0.06,
                'omk': 0.0,
                'tau': 0.054,
                'As': 2.1e-9,
                'ns': 0.9649
            }
        else:
            self.cosmo_fid = cosmo_fid.copy()

        # Compute derived parameters
        if 'Omega_m' not in self.cosmo_fid:
            self.cosmo_fid['Omega_m'] = (
                (self.cosmo_fid['ombh2'] + self.cosmo_fid['omch2']) /
                (self.cosmo_fid['H0'] / 100) ** 2
            )
        if 'sigma8' not in self.cosmo_fid:
            self.cosmo_fid['sigma8'] = 0.8111  # Planck 2018 TT,TE,EE+lowE+lensing

        # Setup lens distribution
        self.lens_file = lens_file
        self._setup_lens_distribution()
        self.theta_masks = {}

        # Precompute Ogata quadrature nodes/weights for J0, J2, J4 Hankel
        # transforms (constants; computed once with scipy, used by JAX).
        self._ogata = {nu: self._ogata_nodes(nu) for nu in (0, 2, 4)}

    @staticmethod
    def _ogata_nodes(nu: int, N: int = OGATA_N, h: float = OGATA_H):
        """
        Nodes x_k and combined weights W_k for Ogata (2005) quadrature:

            int_0^inf g(x) J_nu(x) dx  ~=  sum_k W_k g(x_k)

        with x_k = (pi/h) psi(h r_k), r_k = (k-th zero of J_nu)/pi,
        psi(t) = t tanh((pi/2) sinh t), and
        W_k = pi * [Y_nu(pi r_k)/J_{nu+1}(pi r_k)] * J_nu(x_k) * psi'(h r_k).

        This is the same scheme (and N, h) as the `hankel` package loscov
        uses to build the data, chosen because the nodes sit at Bessel zeros
        so the oscillating lobes cancel by construction - a plain trapezoid
        rule cannot resolve the strong cancellation at large theta.
        """
        from scipy.special import jn_zeros, jv, yv

        zeros = jn_zeros(nu, N)
        r = zeros / np.pi
        t = h * r
        u = 0.5 * np.pi * np.sinh(t)
        psi = t * np.tanh(u)
        # psi'(t); sech^2(u) underflows to exactly 0 for large u, which is fine
        sech2 = 1.0 / np.cosh(np.minimum(u, 350.0)) ** 2
        sech2 = np.where(u < 350.0, sech2, 0.0)
        dpsi = np.tanh(u) + t * 0.5 * np.pi * np.cosh(t) * sech2

        x = (np.pi / h) * psi
        W = np.pi * (yv(nu, np.pi * r) / jv(nu + 1, np.pi * r)) * jv(nu, x) * dpsi

        return jnp.array(x), jnp.array(W)

    def _setup_lens_distribution(self):
        """Load lens catalog from file or use fallback."""
        if self.lens_file is not None:
            lens_file = Path(self.lens_file)
            if lens_file.exists():
                lens_data = np.loadtxt(lens_file, comments='#')
                self.z_d_array = lens_data[:, 0]
                self.z_s_array = lens_data[:, 1]
                self.N_lenses = len(self.z_d_array)
                print(f"Loaded {self.N_lenses} lenses from {lens_file}")
                return
            else:
                warnings.warn(f"Lens file not found: {lens_file}, using fallback")

        # Fallback lens sample
        warnings.warn("Using fallback lens sample (5 lenses)")
        self.z_d_array = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        self.z_s_array = np.array([1.2, 1.5, 1.8, 2.0, 2.3])
        self.N_lenses = len(self.z_d_array)

    def setup_Pk_grid(self, Om_range: Tuple[float, float] = (0.25, 0.40),
                      As_range: Tuple[float, float] = (1.5e-9, 2.7e-9),
                      nOm: int = 5, nAs: int = 5, nz: int = 50, nk: int = 100,
                      verbose: bool = True):
        """
        Pre-compute CAMB P(k) on a grid of cosmologies (varying Omega_m, A_s).

        This is the computationally expensive part - done once at initialization.

        Args:
            Om_range: (min, max) range for Omega_m
            As_range: (min, max) range for A_s
            nOm: Number of Omega_m grid points
            nAs: Number of A_s grid points
            nz: Number of redshift grid points
            nk: Number of wavenumber grid points
            verbose: Print progress messages
        """
        if verbose:
            print("\nPre-computing CAMB P(k) grid...")
            print(f"  Omega_m range: {Om_range}, {nOm} points")
            print(f"  A_s range: {As_range}, {nAs} points")

        Om_grid = np.linspace(*Om_range, nOm)
        As_grid = np.linspace(*As_range, nAs)
        z_grid = np.linspace(Z_MIN_CAMB, Z_MAX_CAMB, nz)
        k_grid = np.logspace(np.log10(K_MIN_CAMB), np.log10(K_MAX_CAMB), nk)

        # Pre-allocate
        Pk_grid = np.zeros((nOm, nAs, nz, nk))
        sigma8_grid = np.zeros((nOm, nAs))

        for i, Om in enumerate(Om_grid):
            for j, As in enumerate(As_grid):
                # Setup CAMB for this cosmology
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    H0=self.cosmo_fid['H0'],
                    ombh2=self.cosmo_fid['ombh2'],
                    omch2=Om * (self.cosmo_fid['H0'] / 100) ** 2 - self.cosmo_fid['ombh2'],
                    mnu=self.cosmo_fid['mnu'],
                    omk=self.cosmo_fid['omk'],
                    tau=self.cosmo_fid['tau']
                )

                pars.InitPower.set_params(As=As, ns=self.cosmo_fid['ns'])
                pars.set_matter_power(redshifts=z_grid.tolist(), kmax=10.0)
                pars.NonLinear = model.NonLinear_both

                results = camb.get_results(pars)

                # Record resulting sigma8 at z=0 for this (Omega_m, A_s)
                s8_camb = results.get_sigma8()[-1]
                sigma8_grid[i, j] = s8_camb

                # Get P(k) for all z
                PK_interp = results.get_matter_power_interpolator(
                    nonlinear=True, hubble_units=False, k_hunit=False
                )
                for iz, z in enumerate(z_grid):
                    Pk_grid[i, j, iz, :] = PK_interp.P(z, k_grid)

                if verbose:
                    print(f"    Computed ({i + 1}/{nOm}, {j + 1}/{nAs}): "
                          f"Omega_m={Om:.3f}, A_s={As:.3e}, sigma_8={s8_camb:.3f}")

        # Store grids as JAX arrays on GPU
        self.Om_grid = jnp.array(Om_grid)
        self.As_grid = jnp.array(As_grid)
        self.z_grid = jnp.array(z_grid)
        self.k_grid = jnp.array(k_grid)
        self.Pk_grid = jnp.array(Pk_grid)
        self.sigma8_grid = jnp.array(sigma8_grid)

        if verbose:
            print("  P(k) grid ready on GPU!")
            print("  sigma_8 mapping stored for each (Omega_m, A_s) grid point")

        # Also setup background cosmology interpolators
        self._setup_background(Om_grid, verbose=verbose)

    def _setup_background(self, Om_grid, verbose: bool = True,
                          n_Om_min: int = 101):
        """Setup background cosmology table chi(z) over Omega_m grid.

        Uses a denser Omega_m grid than the P(k) grid: background CAMB calls
        are cheap, and the kernel geometry tables (and their autodiff
        Om-derivatives) inherit this grid's resolution. Autodiff returns the
        local slope of the piecewise-linear interpolation, so the grid spacing
        directly sets the noise level of dD/dOm; 101 points over the default
        Om range keeps this at the few-percent level.
        """
        n_Om = max(len(Om_grid), n_Om_min)
        Om_grid = np.linspace(Om_grid[0], Om_grid[-1], n_Om)

        z_arr = np.linspace(0, 7, 500)
        chi_table = np.zeros((len(Om_grid), len(z_arr)))

        for i, Om in enumerate(Om_grid):
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=self.cosmo_fid['H0'],
                ombh2=self.cosmo_fid['ombh2'],
                omch2=Om * (self.cosmo_fid['H0'] / 100) ** 2 - self.cosmo_fid['ombh2'],
                mnu=self.cosmo_fid['mnu'],
                omk=self.cosmo_fid['omk'],
            )
            bg = camb.get_background(pars)
            chi_table[i, :] = bg.comoving_radial_distance(z_arr)

        self.z_bg = jnp.array(z_arr)
        self.Om_bg = jnp.array(Om_grid)
        self.chi_bg_table = jnp.array(chi_table)

        if verbose:
            print("Background cosmology table ready on GPU")

    @partial(jit, static_argnums=(0,))
    def chi_of_z(self, Om, z):
        """Comoving distance (Mpc) as function of redshift with Omega_m dependence."""
        iOm = (Om - self.Om_bg[0]) / (self.Om_bg[-1] - self.Om_bg[0]) * (len(self.Om_bg) - 1)
        iOm = jnp.clip(iOm, 0.0, len(self.Om_bg) - 1.001)
        Om_low = jnp.floor(iOm).astype(int)
        Om_high = jnp.minimum(Om_low + 1, len(self.Om_bg) - 1)
        t_Om = iOm - Om_low
        chi_row = (1.0 - t_Om) * self.chi_bg_table[Om_low, :] + t_Om * self.chi_bg_table[Om_high, :]
        return jnp.interp(z, self.z_bg, chi_row)

    @partial(jit, static_argnums=(0,))
    def z_of_chi(self, Om, chi):
        """Redshift as function of comoving distance with Omega_m dependence."""
        iOm = (Om - self.Om_bg[0]) / (self.Om_bg[-1] - self.Om_bg[0]) * (len(self.Om_bg) - 1)
        iOm = jnp.clip(iOm, 0.0, len(self.Om_bg) - 1.001)
        Om_low = jnp.floor(iOm).astype(int)
        Om_high = jnp.minimum(Om_low + 1, len(self.Om_bg) - 1)
        t_Om = iOm - Om_low
        chi_row = (1.0 - t_Om) * self.chi_bg_table[Om_low, :] + t_Om * self.chi_bg_table[Om_high, :]
        return jnp.interp(chi, chi_row, self.z_bg)

    @partial(jit, static_argnums=(0,))
    def Pk_interp(self, Om, s8, z, k):
        """
        Interpolate P(k) from pre-computed grid - JAX multilinear interpolation.

        k beyond the grid is power-law extrapolated (linear continuation in
        log P vs log k, matching loscov's extrap_kmax behaviour). Clamping to
        the edge value instead injects spurious non-decaying small-scale power
        into high-ell C_ells, which swamps the correlation functions at large
        theta. z and (Om, s8) outside the grid still clamp to the edge.

        Args:
            Om, s8: Cosmological parameters
            z, k: Redshift and wavenumber
        """
        from jax.scipy.ndimage import map_coordinates

        # Normalize Omega_m grid coordinate [0, nOm-1]
        iOm = (Om - self.Om_grid[0]) / (self.Om_grid[-1] - self.Om_grid[0]) * (len(self.Om_grid) - 1)
        iOm = jnp.clip(iOm, 0.0, len(self.Om_grid) - 1.001)

        # Interpolate sigma8(Om, A_s) along the A_s axis to find the matching A_s index
        Om_low = jnp.floor(iOm).astype(int)
        Om_high = jnp.minimum(Om_low + 1, len(self.Om_grid) - 1)
        t_Om = iOm - Om_low

        sigma8_row = (1.0 - t_Om) * self.sigma8_grid[Om_low, :] + t_Om * self.sigma8_grid[Om_high, :]

        a_indices = jnp.arange(len(self.As_grid))
        iAs = jnp.interp(s8, sigma8_row, a_indices)
        iAs = jnp.clip(iAs, 0.0, len(self.As_grid) - 1.001)

        # Remaining coordinates (z, k)
        nk = len(self.k_grid)
        iz = (z - self.z_grid[0]) / (self.z_grid[-1] - self.z_grid[0]) * (len(self.z_grid) - 1)
        ik_raw = jnp.log(k / self.k_grid[0]) / jnp.log(self.k_grid[-1] / self.k_grid[0]) * (nk - 1)
        iz = jnp.clip(iz, 0.0, len(self.z_grid) - 1.001)
        ik = jnp.clip(ik_raw, 0.0, nk - 1.001)

        def P_at(ik_val):
            coords = jnp.array([[iOm], [iAs], [iz], [ik_val]])
            return map_coordinates(self.Pk_grid, coords, order=1, mode='nearest')[0]

        P_in = P_at(ik)

        # Power-law extrapolation beyond the k grid: the grid is log-spaced,
        # so one index step is a constant step in log k, and the local
        # log-log slope at the edge continues the spectrum.
        i_hi = nk - 1.001
        logP_hi = jnp.log(P_at(jnp.asarray(i_hi)))
        slope_hi = logP_hi - jnp.log(P_at(jnp.asarray(i_hi - 1.0)))
        P_hi = jnp.exp(logP_hi + slope_hi * (ik_raw - i_hi))

        logP_lo = jnp.log(P_at(jnp.asarray(0.0)))
        slope_lo = jnp.log(P_at(jnp.asarray(1.0))) - logP_lo
        P_lo = jnp.exp(logP_lo + slope_lo * ik_raw)

        return jnp.where(ik_raw > i_hi, P_hi,
                         jnp.where(ik_raw < 0.0, P_lo, P_in))

    def setup_galaxy_distributions(self, data_dir: str, verbose: bool = True):
        """
        Load galaxy redshift distributions from data directory.

        Args:
            data_dir: Path to dataset directory
            verbose: Print progress messages
        """
        data_dir = Path(data_dir)
        zdist_file = data_dir / 'redshift_distributions'

        with open(zdist_file, 'rb') as f:
            zdist = pickle.load(f)

        self.E_dist = zdist['E']
        self.P_dist = zdist['P']
        self.Nbinz_E = int(self.E_dist.Nbinz)
        self.Nbinz_P = int(self.P_dist.Nbinz)
        self.Nbinz = self.Nbinz_E

        if self.Nbinz_E != self.Nbinz_P:
            warnings.warn(
                f"E/P bin mismatch detected (E={self.Nbinz_E}, P={self.Nbinz_P}). "
                "EP uses the overlapping bin range.",
                RuntimeWarning,
            )

        if verbose:
            print(f"Loaded {self.Nbinz_E} E bins and {self.Nbinz_P} P bins")

        if not hasattr(self, 'chi_bg_table'):
            raise RuntimeError("Background table missing. Call setup_Pk_grid() before setup_galaxy_distributions().")

        # Pre-compute distribution tables and kernels on fixed grids.
        self._precompute_distribution_pdfs(verbose=verbose)
        self._precompute_QL_mean(verbose=verbose)
        self._precompute_QE_mean(verbose=verbose)

    def _evaluate_bin_pdf(self, dist, bin_idx: int, z_grid: np.ndarray) -> np.ndarray:
        """Evaluate and normalize a tomographic-bin redshift PDF on a fixed z-grid."""
        z_min = float(dist.limits[bin_idx])
        z_max = float(dist.limits[bin_idx + 1])

        if hasattr(dist, 'pb') and callable(dist.pb):
            pdf = np.array([float(dist.pb(float(z), bin_idx)) for z in z_grid], dtype=float)
        else:
            # Fallback for minimal shim objects: uniform within the bin.
            in_bin = (z_grid >= z_min) & (z_grid <= z_max)
            width = max(z_max - z_min, 1e-6)
            pdf = in_bin.astype(float) / width

        pdf = np.clip(pdf, 0.0, None)
        norm = float(np.trapezoid(pdf, z_grid))

        if not np.isfinite(norm) or norm <= 0.0:
            # Robust fallback so setup remains usable with incomplete legacy pickles.
            z_mid = 0.5 * (z_min + z_max)
            sigma = max(0.25 * (z_max - z_min), 1e-3)
            pdf = np.exp(-0.5 * ((z_grid - z_mid) / sigma) ** 2)
            norm = float(np.trapezoid(pdf, z_grid))

        return pdf / max(norm, 1e-12)

    def _precompute_distribution_pdfs(self, nz: int = N_Z_KERNEL, verbose: bool = True):
        """Pre-compute redshift PDFs for E/P bins on a common z-grid."""
        z_max = max(float(self.E_dist.limits[-1]), float(self.P_dist.limits[-1]), float(self.z_bg[-1]))
        z_grid_np = np.linspace(0.0, z_max, nz)

        E_pdf_np = np.zeros((self.Nbinz_E, nz))
        P_pdf_np = np.zeros((self.Nbinz_P, nz))

        for b in range(self.Nbinz_E):
            E_pdf_np[b, :] = self._evaluate_bin_pdf(self.E_dist, b, z_grid_np)
        for b in range(self.Nbinz_P):
            P_pdf_np[b, :] = self._evaluate_bin_pdf(self.P_dist, b, z_grid_np)

        self.z_pdf_grid = jnp.array(z_grid_np)
        self.E_pdf_table = jnp.array(E_pdf_np)
        self.P_pdf_table = jnp.array(P_pdf_np)

        if verbose:
            print(f"Pre-computed E/P redshift PDFs on {nz} z-points")

    def _precompute_QL_mean(self, chi_min: float = CHI_MIN_DEFAULT,
                            chi_max: float = CHI_MAX_QL_GRID,
                            nchi: int = N_CHI_QL, verbose: bool = True):
        """Pre-compute mean LOS geometry kernel K_LOS(chi), averaged over lenses.

        Tabulated on the Omega_m grid so the geometric response (lens/source
        distances shifting with cosmology) survives autodiff; a single
        fiducial-Om table would freeze dK/dOm to zero and bias the Fisher
        derivatives.
        """
        chi_grid_np = np.linspace(chi_min, chi_max, nchi)
        Om_grid_np = np.array(self.Om_bg)
        KL_mean_np = np.zeros((len(Om_grid_np), nchi))

        if verbose:
            print("Pre-computing mean LOS kernel on Omega_m grid...")

        chi_col = chi_grid_np[:, None]  # (nchi, 1) vs lens arrays (1, N_lenses)
        for iOm, Om in enumerate(Om_grid_np):
            chi_d_row = np.array(self.chi_of_z(Om, jnp.array(self.z_d_array)))[None, :]
            chi_s_row = np.array(self.chi_of_z(Om, jnp.array(self.z_s_array)))[None, :]
            # Strong-lensing LOS convergence weight, matching loscov
            # (functions/correlations/LL.py): K = os + od - ds, each term
            # active only where positive. Vanishes behind the source.
            os_w = (chi_s_row - chi_col) / chi_s_row
            od_w = (chi_d_row - chi_col) / chi_d_row
            ds_w = ((chi_col - chi_d_row) * (chi_s_row - chi_col)
                    / (chi_col * (chi_s_row - chi_d_row)))
            K_vals = (os_w * np.heaviside(os_w, 0.0)
                      + od_w * np.heaviside(od_w, 0.0)
                      - ds_w * np.heaviside(ds_w, 0.0))
            KL_mean_np[iOm, :] = K_vals.mean(axis=1)

        # Transfer to JAX/GPU
        self.chi_QL_grid = jnp.array(chi_grid_np)
        self.KL_mean_grid = jnp.array(KL_mean_np)

        if verbose:
            print(f"  Mean LOS kernel ready on GPU "
                  f"({len(Om_grid_np)} Om points x {nchi} chi points)")

    def _precompute_QE_mean(self, chi_min: float = CHI_MIN_DEFAULT,
                            chi_max: float = CHI_MAX_QL_GRID,
                            nchi: int = N_CHI_QL, verbose: bool = True):
        """Pre-compute mean weak-lensing geometry kernel K_E(chi) for each E bin.

        Tabulated on the Omega_m grid (see _precompute_QL_mean) so that the
        source-distance response to cosmology survives autodiff.
        """
        chi_grid_np = np.linspace(chi_min, chi_max, nchi)
        Om_grid_np = np.array(self.Om_bg)
        KE_mean_np = np.zeros((len(Om_grid_np), self.Nbinz_E, nchi))

        if verbose:
            print("Pre-computing mean E-kernels on Omega_m grid...")

        z_pdf_np = np.array(self.z_pdf_grid)
        E_pdf_np = np.array(self.E_pdf_table)
        chi_col = chi_grid_np[:, None]

        for iOm, Om in enumerate(Om_grid_np):
            chi_source = np.array(self.chi_of_z(Om, jnp.array(z_pdf_np)))
            chi_src_row = np.maximum(chi_source, 1e-6)[None, :]
            K_matrix = np.where(chi_col < chi_src_row,
                                (chi_src_row - chi_col) / chi_src_row, 0.0)

            for b in range(self.Nbinz_E):
                KE_mean_np[iOm, b, :] = np.trapezoid(
                    K_matrix * E_pdf_np[b][None, :], z_pdf_np, axis=1)

        self.chi_QE_grid = jnp.array(chi_grid_np)
        self.KE_mean_grid = jnp.array(KE_mean_np)

        if verbose:
            print(f"  Mean E-kernels ready on GPU ({len(Om_grid_np)} Om points, "
                  f"{self.Nbinz_E} bins, {nchi} chi points)")

    @partial(jit, static_argnums=(0,))
    def _Om_interp_row(self, table, Om):
        """Linear interpolation of a table's leading Omega_m axis."""
        iOm = (Om - self.Om_bg[0]) / (self.Om_bg[-1] - self.Om_bg[0]) * (len(self.Om_bg) - 1)
        iOm = jnp.clip(iOm, 0.0, len(self.Om_bg) - 1.001)
        Om_low = jnp.floor(iOm).astype(int)
        Om_high = jnp.minimum(Om_low + 1, len(self.Om_bg) - 1)
        t_Om = iOm - Om_low
        return (1.0 - t_Om) * table[Om_low] + t_Om * table[Om_high]

    @partial(jit, static_argnums=(0,))
    def QL_mean(self, chi, Om):
        """Mean LOS lensing kernel with Omega_m-dependent geometry."""
        K_row = self._Om_interp_row(self.KL_mean_grid, Om)
        K = jnp.interp(chi, self.chi_QL_grid, K_row)
        z = self.z_of_chi(Om, chi)
        prefactor = -(3.0 / 2.0) * Om * (self.cosmo_fid['H0'] ** 2) / (self.c_km_s ** 2)
        return prefactor * (1 + z) * K

    @partial(jit, static_argnums=(0,))
    def galaxy_bias(self, z):
        """Redshift-dependent galaxy bias from Euclid recipe."""
        return 1.1 * z ** 2.4 / (1 + z) + 0.9

    @partial(jit, static_argnums=(0,))
    def QE_mean(self, Om, chi, bin_idx):
        """Mean Q_E kernel averaged over source redshift distribution in one E bin."""
        K_table = jnp.take(self.KE_mean_grid, bin_idx, axis=1)  # (nOm, nchi)
        K_row = self._Om_interp_row(K_table, Om)
        K = jnp.interp(chi, self.chi_QE_grid, K_row)
        z = self.z_of_chi(Om, chi)
        prefactor = -(3.0 / 2.0) * Om * (self.cosmo_fid['H0'] ** 2) / (self.c_km_s ** 2)
        return prefactor * (1 + z) * K

    @partial(jit, static_argnums=(0,))
    def QP_mean(self, Om, chi, bin_idx):
        """Mean Q_P kernel for one P tomographic bin."""
        z = self.z_of_chi(Om, chi)
        pz = jnp.interp(z, self.z_pdf_grid, self.P_pdf_table[bin_idx], left=0.0, right=0.0)
        Ez = jnp.sqrt(jnp.maximum(Om * (1 + z) ** 3 + (1.0 - Om), 1e-12))
        H_over_c = (self.cosmo_fid['H0'] / self.c_km_s) * Ez
        return H_over_c * pz * self.galaxy_bias(z) / jnp.maximum(chi, 1e-6)

    @partial(jit, static_argnums=(0,))
    def compute_Cl_LL_jax(self, Om, s8, ell, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """Compute C_ell^LL using JAX - fully differentiable."""
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QL = self.QL_mean(chi_grid, Om)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)

        integrand = QL * QL * Pk
        Cl = jnp.trapezoid(integrand, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_LE_jax(self, Om, s8, ell, bin_idx, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """Compute C_ell^LE using mean Q_L and distribution-averaged Q_E kernels."""
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QL = self.QL_mean(chi_grid, Om)
        QE = self.QE_mean(Om, chi_grid, bin_idx)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)

        integrand = QL * QE * Pk
        Cl = jnp.trapezoid(integrand, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_LP_jax(self, Om, s8, ell, bin_idx, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """Compute C_ell^LP using mean Q_L and distribution-averaged Q_P kernels."""
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QL = self.QL_mean(chi_grid, Om)
        QP = self.QP_mean(Om, chi_grid, bin_idx)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)

        integrand = QL * QP * Pk
        Cl = jnp.trapezoid(integrand, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_EE_jax(self, Om, s8, ell, bin_idx, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """
        Compute C_ell^EE - cosmic shear auto-correlation.

        Uses distribution-averaged Q_E × Q_E kernels.
        """
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QE = self.QE_mean(Om, chi_grid, bin_idx)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)

        integrand = QE * QE * Pk
        Cl = jnp.trapezoid(integrand, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_EP_jax(self, Om, s8, ell, bin_idx, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """
        Compute C_ell^EP - cosmic shear × position cross-correlation.

        Uses distribution-averaged Q_E and Q_P kernels.
        """
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QE = self.QE_mean(Om, chi_grid, bin_idx)
        QP = self.QP_mean(Om, chi_grid, bin_idx)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)
        Cl = jnp.trapezoid(QE * QP * Pk, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_PP_jax(self, Om, s8, ell, bin_idx, chi_min=CHI_MIN_DEFAULT,
                          chi_max=CHI_MAX_DEFAULT, nchi=N_CHI_CL):
        """
        Compute C_ell^PP - galaxy clustering auto-correlation.

        Uses distribution-averaged Q_P × Q_P kernels.
        """
        chi_grid = jnp.linspace(chi_min, chi_max, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QP = self.QP_mean(Om, chi_grid, bin_idx)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)
        Cl = jnp.trapezoid(QP * QP * Pk, chi_grid)
        return Cl

    @partial(jit, static_argnums=(0,))
    def bessel_j0(self, x):
        """
        Bessel function of the first kind, order 0.

        Uses JAX's native bessel_jn for all x values - accurate and differentiable.
        """
        from jax.scipy.special import bessel_jn
        return bessel_jn(x, v=0)[0]

    @partial(jit, static_argnums=(0,))
    def bessel_j2(self, x):
        """
        Bessel function of the first kind, order 2.

        Uses JAX's native bessel_jn for all x values - accurate and differentiable.
        """
        from jax.scipy.special import bessel_jn
        return bessel_jn(x, v=2)[2]

    @partial(jit, static_argnums=(0,))
    def bessel_j4(self, x):
        """
        Bessel function of the first kind, order 4.

        Uses JAX's native bessel_jn for all x values - accurate and differentiable.
        """
        from jax.scipy.special import bessel_jn
        return bessel_jn(x, v=4)[4]

    @partial(jit, static_argnums=(0,))
    def _cl_eval(self, Cl_vals, ell_grid, ell_query):
        """
        Evaluate C(ell) at arbitrary multipoles from a computed grid.

        Inside the grid: linear interpolation in log(ell).
        Outside: power-law continuation using the log-log slope at the grid
        edge (scaling the signed edge value, so negative spectra work).
        """
        log_grid = jnp.log(ell_grid)
        log_q = jnp.log(ell_query)
        C_in = jnp.interp(log_q, log_grid, Cl_vals)

        tiny = 1e-300
        # high-ell power-law tail
        slope_hi = ((jnp.log(jnp.abs(Cl_vals[-1]) + tiny)
                     - jnp.log(jnp.abs(Cl_vals[-2]) + tiny))
                    / (log_grid[-1] - log_grid[-2]))
        C_hi = Cl_vals[-1] * jnp.exp(slope_hi * (log_q - log_grid[-1]))
        # low-ell power-law tail
        slope_lo = ((jnp.log(jnp.abs(Cl_vals[1]) + tiny)
                     - jnp.log(jnp.abs(Cl_vals[0]) + tiny))
                    / (log_grid[1] - log_grid[0]))
        C_lo = Cl_vals[0] * jnp.exp(slope_lo * (log_q - log_grid[0]))

        return jnp.where(log_q > log_grid[-1], C_hi,
                         jnp.where(log_q < log_grid[0], C_lo, C_in))

    @partial(jit, static_argnums=(0,))
    def _hankel_ogata(self, Cl_func_values, ell_grid, theta, x_nodes, W_nodes):
        """
        xi(theta) = 1/(2 pi) int ell C(ell) J_nu(ell theta) d ell,
        computed with Ogata quadrature: substituting x = ell*theta,
        xi(theta) = 1/(2 pi theta^2) sum_k W_k x_k C(x_k / theta).

        The Bessel factor J_nu(x_k) is already inside W_k (piece 1), and
        C is evaluated anywhere via _cl_eval (piece 2), so this is an
        accurate oscillatory integral yet still a plain differentiable sum.
        """
        theta = jnp.asarray(theta)
        scalar_theta = theta.ndim == 0
        theta_vec = theta[None] if scalar_theta else theta

        ell_q = x_nodes[:, None] / theta_vec[None, :]          # (N, n_theta)
        C_q = self._cl_eval(Cl_func_values, ell_grid, ell_q)
        xi = (W_nodes[:, None] * x_nodes[:, None] * C_q).sum(axis=0)
        xi = xi / (2.0 * jnp.pi * theta_vec ** 2)
        return xi[0] if scalar_theta else xi

    def hankel_j0(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J0 Bessel function (Ogata quadrature).

        Args:
            Cl_func_values: C_ell values on ell_grid, shape (n_ell,)
            ell_grid: ell values, shape (n_ell,)
            theta: scalar theta or array, shape (n_theta,)

        Returns:
            xi(theta) as a scalar (if theta is scalar) or array (n_theta,).
        """
        return self._hankel_ogata(Cl_func_values, ell_grid, theta, *self._ogata[0])

    def hankel_j2(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J2 Bessel function (Ogata quadrature)."""
        return self._hankel_ogata(Cl_func_values, ell_grid, theta, *self._ogata[2])

    def hankel_j4(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J4 Bessel function (Ogata quadrature)."""
        return self._hankel_ogata(Cl_func_values, ell_grid, theta, *self._ogata[4])

    def load_angular_bins(self, data_dir: str):
        """
        Load angular bin information.

        Args:
            data_dir: Path to dataset directory
        """
        data_dir = Path(data_dir)
        ang_file = data_dir / 'angular_distributions'

        with open(ang_file, 'rb') as f:
            ang_dist = pickle.load(f)

        # Representative angles (bin centroids) - kept for scale cuts/reporting
        self.theta_LL_plus = jnp.array(ang_dist['LL_plus'].Thetas)
        self.theta_LL_minus = jnp.array(ang_dist['LL_minus'].Thetas)

        # Bin-averaging nodes/weights (the data vector is annulus-averaged)
        self.nodes_LL_plus, self.w_LL_plus = self._bin_average_nodes(ang_dist['LL_plus'])
        self.nodes_LL_minus, self.w_LL_minus = self._bin_average_nodes(ang_dist['LL_minus'])

        # Dynamically detect number of tomographic bins for E/P probes.
        self.n_tomo_bins_E = len(ang_dist['LE_plus'])
        self.n_tomo_bins_P = len(ang_dist['LP'])
        self.n_tomo_bins = self.n_tomo_bins_E
        self.theta_LE_plus = [jnp.array(ang_dist['LE_plus'][i].Thetas) for i in range(self.n_tomo_bins_E)]
        self.theta_LE_minus = [jnp.array(ang_dist['LE_minus'][i].Thetas) for i in range(self.n_tomo_bins_E)]
        self.theta_LP = [jnp.array(ang_dist['LP'][i].Thetas) for i in range(self.n_tomo_bins_P)]

        le_plus_nw = [self._bin_average_nodes(ang_dist['LE_plus'][i]) for i in range(self.n_tomo_bins_E)]
        le_minus_nw = [self._bin_average_nodes(ang_dist['LE_minus'][i]) for i in range(self.n_tomo_bins_E)]
        lp_nw = [self._bin_average_nodes(ang_dist['LP'][i]) for i in range(self.n_tomo_bins_P)]
        self.nodes_LE_plus = [n for n, _ in le_plus_nw]
        self.w_LE_plus = [w for _, w in le_plus_nw]
        self.nodes_LE_minus = [n for n, _ in le_minus_nw]
        self.w_LE_minus = [w for _, w in le_minus_nw]
        self.nodes_LP = [n for n, _ in lp_nw]
        self.w_LP = [w for _, w in lp_nw]

        # EE uses same structure as LE (plus/minus for cosmic shear)
        self.theta_EE_plus = self.theta_LE_plus  # Reuse LE plus bins
        self.theta_EE_minus = self.theta_LE_minus  # Reuse LE minus bins

        # PP uses same structure as LP (galaxy clustering)
        self.theta_PP = self.theta_LP  # Reuse LP bins

        # EP uses LP structure (no plus/minus for cross-correlation)
        self.theta_EP = self.theta_LP  # Reuse LP bins

        # Corresponding bin-average nodes/weights
        self.nodes_EE_plus, self.w_EE_plus = self.nodes_LE_plus, self.w_LE_plus
        self.nodes_EE_minus, self.w_EE_minus = self.nodes_LE_minus, self.w_LE_minus
        self.nodes_PP, self.w_PP = self.nodes_LP, self.w_LP
        self.nodes_EP, self.w_EP = self.nodes_LP, self.w_LP

    def apply_theta_min_cut(self, theta_min_arcmin: Optional[float] = None):
        """
        Apply a minimum-angle cut (in arcmin) to all angular bins.

        Args:
            theta_min_arcmin: Minimum theta in arcminutes; bins below are dropped.
        """
        if theta_min_arcmin is None:
            self.theta_min_rad = None
            self.theta_masks = {}
            self.theta_cut_report = {}
            return

        theta_min_rad = theta_min_arcmin * np.pi / 180.0 / 60.0
        self.theta_min_rad = theta_min_rad

        self.theta_masks = {}
        self.theta_cut_report = {}

        def _filter(arr, key):
            arr_np = np.array(arr)
            mask = arr_np >= theta_min_rad
            filtered = arr_np[mask]
            if filtered.size == 0:
                raise ValueError(f"All theta values were cut (theta_min={theta_min_arcmin} arcmin)")
            self.theta_masks[key] = mask
            self.theta_cut_report[key] = {'before': len(arr_np), 'after': len(filtered)}
            return jnp.array(filtered)

        def _filter_nodes(nodes, weights, key):
            mask = self.theta_masks[key]
            w_np = np.array(weights)
            n_np = np.array(nodes).reshape(w_np.shape)
            return jnp.array(n_np[mask].ravel()), jnp.array(w_np[mask])

        self.theta_LL_plus = _filter(self.theta_LL_plus, 'LL_plus')
        self.theta_LL_minus = _filter(self.theta_LL_minus, 'LL_minus')
        self.theta_LE_plus = [_filter(arr, f'LE_plus_{i}') for i, arr in enumerate(self.theta_LE_plus)]
        self.theta_LE_minus = [_filter(arr, f'LE_minus_{i}') for i, arr in enumerate(self.theta_LE_minus)]
        self.theta_LP = [_filter(arr, f'LP_{i}') for i, arr in enumerate(self.theta_LP)]

        # Apply the same per-bin masks to the bin-averaging nodes/weights
        self.nodes_LL_plus, self.w_LL_plus = _filter_nodes(
            self.nodes_LL_plus, self.w_LL_plus, 'LL_plus')
        self.nodes_LL_minus, self.w_LL_minus = _filter_nodes(
            self.nodes_LL_minus, self.w_LL_minus, 'LL_minus')
        for i in range(len(self.theta_LE_plus)):
            self.nodes_LE_plus[i], self.w_LE_plus[i] = _filter_nodes(
                self.nodes_LE_plus[i], self.w_LE_plus[i], f'LE_plus_{i}')
            self.nodes_LE_minus[i], self.w_LE_minus[i] = _filter_nodes(
                self.nodes_LE_minus[i], self.w_LE_minus[i], f'LE_minus_{i}')
        for i in range(len(self.theta_LP)):
            self.nodes_LP[i], self.w_LP[i] = _filter_nodes(
                self.nodes_LP[i], self.w_LP[i], f'LP_{i}')

        # EE, EP, PP automatically filtered since they reference the same arrays
        # Just update the references after filtering
        self.theta_EE_plus = self.theta_LE_plus
        self.theta_EE_minus = self.theta_LE_minus
        self.theta_PP = self.theta_LP
        self.theta_EP = self.theta_LP
        self.nodes_EE_plus, self.w_EE_plus = self.nodes_LE_plus, self.w_LE_plus
        self.nodes_EE_minus, self.w_EE_minus = self.nodes_LE_minus, self.w_LE_minus
        self.nodes_PP, self.w_PP = self.nodes_LP, self.w_LP
        self.nodes_EP, self.w_EP = self.nodes_LP, self.w_LP

    def predict_data_vector_jax(self, Om, s8, ell_grid=None):
        """
        Generate theory predictions for given cosmology using JAX.

        This is the function that gets differentiated for Fisher matrices.

        Args:
            Om: Omega_m value
            s8: sigma_8 value
            ell_grid: Multipoles for Hankel transform (if None, use default)

        Returns:
            Flat array of predictions, shape (n_data,)
        """
        if ell_grid is None:
            ell_grid = jnp.logspace(np.log10(ELL_MIN_DEFAULT), np.log10(ELL_MAX_DEFAULT), N_ELL_DEFAULT)

        predictions_parts = []

        # All entries are annulus averages over the angular bins (the loscov
        # data vector and covariance are bin-averaged, not point-evaluated).

        # LL predictions
        Cl_LL_vals = vmap(lambda ell: self.compute_Cl_LL_jax(Om, s8, ell))(ell_grid)

        xi_LL_plus = self._bin_average(
            self.hankel_j0(Cl_LL_vals, ell_grid, self.nodes_LL_plus), self.w_LL_plus)
        xi_LL_minus = self._bin_average(
            self.hankel_j4(Cl_LL_vals, ell_grid, self.nodes_LL_minus), self.w_LL_minus)
        predictions_parts.append(xi_LL_plus)
        predictions_parts.append(xi_LL_minus)

        n_e_bins = len(self.theta_LE_plus)
        n_p_bins = len(self.theta_LP)

        # LE predictions
        for bin_idx in range(n_e_bins):
            Cl_LE_vals = vmap(lambda ell: self.compute_Cl_LE_jax(Om, s8, ell, bin_idx))(ell_grid)

            xi_LE_plus = self._bin_average(
                self.hankel_j0(Cl_LE_vals, ell_grid, self.nodes_LE_plus[bin_idx]),
                self.w_LE_plus[bin_idx])
            xi_LE_minus = self._bin_average(
                self.hankel_j4(Cl_LE_vals, ell_grid, self.nodes_LE_minus[bin_idx]),
                self.w_LE_minus[bin_idx])
            predictions_parts.append(xi_LE_plus)
            predictions_parts.append(xi_LE_minus)

        # LP predictions
        for bin_idx in range(n_p_bins):
            Cl_LP_vals = vmap(lambda ell: self.compute_Cl_LP_jax(Om, s8, ell, bin_idx))(ell_grid)

            xi_LP = self._bin_average(
                self.hankel_j2(Cl_LP_vals, ell_grid, self.nodes_LP[bin_idx]),
                self.w_LP[bin_idx])
            predictions_parts.append(xi_LP)

        # EE predictions (cosmic shear auto-correlation)
        for bin_idx in range(n_e_bins):
            Cl_EE_vals = vmap(lambda ell: self.compute_Cl_EE_jax(Om, s8, ell, bin_idx))(ell_grid)

            xi_EE_plus = self._bin_average(
                self.hankel_j0(Cl_EE_vals, ell_grid, self.nodes_EE_plus[bin_idx]),
                self.w_EE_plus[bin_idx])
            xi_EE_minus = self._bin_average(
                self.hankel_j4(Cl_EE_vals, ell_grid, self.nodes_EE_minus[bin_idx]),
                self.w_EE_minus[bin_idx])
            predictions_parts.append(xi_EE_plus)
            predictions_parts.append(xi_EE_minus)

        # EP predictions (cosmic shear × position cross-correlation)
        for bin_idx in range(min(n_e_bins, n_p_bins)):
            Cl_EP_vals = vmap(lambda ell: self.compute_Cl_EP_jax(Om, s8, ell, bin_idx))(ell_grid)

            xi_EP = self._bin_average(
                self.hankel_j2(Cl_EP_vals, ell_grid, self.nodes_EP[bin_idx]),
                self.w_EP[bin_idx])
            predictions_parts.append(xi_EP)

        # PP predictions (galaxy clustering auto-correlation)
        # w(theta) is scalar x scalar, so the Hankel transform uses J0
        # (unlike LP/EP, which are spin-2 x scalar and use J2)
        for bin_idx in range(n_p_bins):
            Cl_PP_vals = vmap(lambda ell: self.compute_Cl_PP_jax(Om, s8, ell, bin_idx))(ell_grid)

            xi_PP = self._bin_average(
                self.hankel_j0(Cl_PP_vals, ell_grid, self.nodes_PP[bin_idx]),
                self.w_PP[bin_idx])
            predictions_parts.append(xi_PP)

        return jnp.concatenate(predictions_parts)

    def predict_data_vector_batched(self, Om_batch, s8_batch, ell_grid=None):
        """
        Generate theory predictions for multiple cosmologies in parallel.

        Args:
            Om_batch: Array of Omega_m values, shape (N,)
            s8_batch: Array of sigma_8 values, shape (N,)
            ell_grid: Multipoles for Hankel transform

        Returns:
            Array of predictions, shape (N, n_data)
        """
        predict_fn = vmap(lambda Om, s8: self.predict_data_vector_jax(Om, s8, ell_grid))
        return predict_fn(Om_batch, s8_batch)

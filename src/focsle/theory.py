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
        self.galaxy_bias = 1.5

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
        z_grid = np.linspace(0.0, 3.5, nz)
        k_grid = np.logspace(-4, 1, nk)  # h/Mpc

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
                for iz, z in enumerate(z_grid):
                    PK_interp = results.get_matter_power_interpolator(
                        nonlinear=True, hubble_units=False, k_hunit=False
                    )
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

    def _setup_background(self, Om_grid, verbose: bool = True):
        """Setup background cosmology table chi(z) over Omega_m grid."""
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
        iz = (z - self.z_grid[0]) / (self.z_grid[-1] - self.z_grid[0]) * (len(self.z_grid) - 1)
        ik = jnp.log(k / self.k_grid[0]) / jnp.log(self.k_grid[-1] / self.k_grid[0]) * (len(self.k_grid) - 1)
        iz = jnp.clip(iz, 0.0, len(self.z_grid) - 1.001)
        ik = jnp.clip(ik, 0.0, len(self.k_grid) - 1.001)

        # 4D linear interpolation using map_coordinates
        coords = jnp.array([[iOm], [iAs], [iz], [ik]])
        return map_coordinates(self.Pk_grid, coords, order=1, mode='nearest')[0]

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
        self.Nbinz = self.E_dist.Nbinz

        if verbose:
            print(f"Loaded {self.Nbinz} tomographic bins")

        # Pre-compute mean Q_L kernel on grid
        self._precompute_QL_mean(verbose=verbose)

    def _precompute_QL_mean(self, chi_min: float = 10, chi_max: float = 10000,
                            nchi: int = 200, verbose: bool = True):
        """Pre-compute mean Q_L kernel (numpy, then transfer to JAX)."""
        chi_grid_np = np.linspace(chi_min, chi_max, nchi)
        QL_mean_np = np.zeros(nchi)

        if verbose:
            print("Pre-computing mean Q_L kernel...")

        # Convert lens arrays to numpy for this computation
        Om_fid = self.cosmo_fid['Omega_m']
        chi_d_np = np.array(self.chi_of_z(Om_fid, jnp.array(self.z_d_array)))
        chi_s_np = np.array(self.chi_of_z(Om_fid, jnp.array(self.z_s_array)))

        for j, chi in enumerate(chi_grid_np):
            for i in range(self.N_lenses):
                chi_d, chi_s = chi_d_np[i], chi_s_np[i]

                if chi < chi_d:
                    K = 0.0
                elif chi < chi_s:
                    K = (chi - chi_d) / chi
                else:
                    K = (chi_s - chi_d) / chi

                z_chi = np.interp(chi, np.array(self.chi_bg_table)[0], np.array(self.z_bg))
                prefactor = -(3.0 / 2.0) * self.cosmo_fid['Omega_m'] * \
                           (self.cosmo_fid['H0'] ** 2) / (self.c_km_s ** 2)
                Q_L_val = prefactor * (1 + z_chi) * K

                QL_mean_np[j] += Q_L_val

            QL_mean_np[j] /= self.N_lenses

        # Transfer to JAX/GPU
        self.chi_QL_grid = jnp.array(chi_grid_np)
        self.QL_mean_grid = jnp.array(QL_mean_np)

        if verbose:
            print(f"  Mean Q_L ready on GPU ({nchi} points)")

    @partial(jit, static_argnums=(0,))
    def QL_mean(self, chi, Om):
        """Mean Q_L kernel - JAX interpolation with live Omega_m scaling."""
        base = jnp.interp(chi, self.chi_QL_grid, self.QL_mean_grid)
        return base * (Om / self.cosmo_fid['Omega_m'])

    @partial(jit, static_argnums=(0,))
    def QE(self, Om, chi, chi_star):
        """Q_E kernel for cosmic shear - JAX implementation."""
        z = self.z_of_chi(Om, chi)
        K = jnp.where(chi < chi_star, (chi_star - chi) / chi, 0.0)
        prefactor = -(3.0 / 2.0) * Om * (self.cosmo_fid['H0'] ** 2) / (self.c_km_s ** 2)
        return prefactor * (1 + z) * K

    @partial(jit, static_argnums=(0,))
    def compute_Cl_LL_jax(self, Om, s8, ell, chi_min=10.0, chi_max=8000.0, nchi=100):
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
    def compute_Cl_LE_jax(self, Om, s8, ell, z_gal, chi_min=10.0, chi_max=None, nchi=100):
        """Compute C_ell^LE using JAX."""
        chi_gal = self.chi_of_z(Om, z_gal)
        chi_max_actual = jnp.minimum(8000.0, chi_gal * 1.2) if chi_max is None else chi_max

        chi_grid = jnp.linspace(chi_min, chi_max_actual, nchi)
        z_grid = self.z_of_chi(Om, chi_grid)
        k_grid = (ell + 0.5) / chi_grid

        QL = self.QL_mean(chi_grid, Om)
        QE = self.QE(Om, chi_grid, chi_gal)
        Pk = vmap(lambda z, k: self.Pk_interp(Om, s8, z, k))(z_grid, k_grid)

        integrand = QL * QE * Pk
        Cl = jnp.trapezoid(integrand, chi_grid)

        return Cl

    @partial(jit, static_argnums=(0,))
    def compute_Cl_LP_jax(self, Om, s8, ell, z_gal):
        """Compute C_ell^LP using JAX - delta function means direct evaluation."""
        chi_gal = self.chi_of_z(Om, z_gal)
        k_at_gal = (ell + 0.5) / chi_gal

        QP_coeff = self.galaxy_bias / chi_gal
        QL_at_gal = self.QL_mean(chi_gal, Om)
        Pk_at_gal = self.Pk_interp(Om, s8, z_gal, k_at_gal)

        Cl = QL_at_gal * QP_coeff * Pk_at_gal

        return Cl

    @partial(jit, static_argnums=(0,))
    def bessel_j0(self, x):
        """Bessel J0 - hybrid: series for small x, bessel_jn for large x."""
        from jax.scipy.special import bessel_jn

        def j0_series(x):
            x2 = (x / 2.0) ** 2
            result = 1.0 - x2 * (1.0 - x2 / 4.0 * (1.0 - x2 / 9.0 * (1.0 - x2 / 16.0)))
            return result

        return jnp.where(jnp.abs(x) < 5.0, j0_series(x), bessel_jn(x, v=0)[0])

    @partial(jit, static_argnums=(0,))
    def bessel_j2(self, x):
        """Bessel J2 - hybrid: series for small x, bessel_jn for large x."""
        from jax.scipy.special import bessel_jn

        def j2_series(x):
            x_half = x / 2.0
            x2 = x_half * x_half
            result = x2 / 2.0 * (1.0 - x2 / 6.0 * (1.0 - x2 / 10.0 * (1.0 - x2 / 14.0)))
            return result

        return jnp.where(jnp.abs(x) < 5.0, j2_series(x), bessel_jn(x, v=2)[2])

    @partial(jit, static_argnums=(0,))
    def bessel_j4(self, x):
        """Bessel J4 - hybrid: series for small x, bessel_jn for large x."""
        from jax.scipy.special import bessel_jn

        def j4_series(x):
            x_half = x / 2.0
            x2 = x_half * x_half
            x4 = x2 * x2
            result = x4 / 384.0 * (1.0 - x2 / 10.0 * (1.0 - x2 / 14.0))
            return result

        return jnp.where(jnp.abs(x) < 5.0, j4_series(x), bessel_jn(x, v=4)[4])

    @partial(jit, static_argnums=(0,))
    def hankel_j0(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J0 Bessel function."""
        x_grid = ell_grid * theta
        J0_vals = vmap(self.bessel_j0)(x_grid)
        integrand = ell_grid * Cl_func_values * J0_vals / (2 * jnp.pi)
        return jnp.trapezoid(integrand, ell_grid)

    @partial(jit, static_argnums=(0,))
    def hankel_j2(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J2 Bessel function."""
        x_grid = ell_grid * theta
        J2_vals = vmap(self.bessel_j2)(x_grid)
        integrand = ell_grid * Cl_func_values * J2_vals / (2 * jnp.pi)
        return jnp.trapezoid(integrand, ell_grid)

    @partial(jit, static_argnums=(0,))
    def hankel_j4(self, Cl_func_values, ell_grid, theta):
        """Hankel transform with J4 Bessel function."""
        x_grid = ell_grid * theta
        J4_vals = vmap(self.bessel_j4)(x_grid)
        integrand = ell_grid * Cl_func_values * J4_vals / (2 * jnp.pi)
        return jnp.trapezoid(integrand, ell_grid)

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

        # Store angular grids for each observable
        self.theta_LL_plus = jnp.array(ang_dist['LL_plus'].Thetas)
        self.theta_LL_minus = jnp.array(ang_dist['LL_minus'].Thetas)

        # Dynamically detect number of tomographic bins
        self.n_tomo_bins = len(ang_dist['LE_plus'])
        self.theta_LE_plus = [jnp.array(ang_dist['LE_plus'][i].Thetas) for i in range(self.n_tomo_bins)]
        self.theta_LE_minus = [jnp.array(ang_dist['LE_minus'][i].Thetas) for i in range(self.n_tomo_bins)]
        self.theta_LP = [jnp.array(ang_dist['LP'][i].Thetas) for i in range(self.n_tomo_bins)]

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

        self.theta_LL_plus = _filter(self.theta_LL_plus, 'LL_plus')
        self.theta_LL_minus = _filter(self.theta_LL_minus, 'LL_minus')
        self.theta_LE_plus = [_filter(arr, f'LE_plus_{i}') for i, arr in enumerate(self.theta_LE_plus)]
        self.theta_LE_minus = [_filter(arr, f'LE_minus_{i}') for i, arr in enumerate(self.theta_LE_minus)]
        self.theta_LP = [_filter(arr, f'LP_{i}') for i, arr in enumerate(self.theta_LP)]

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
            ell_grid = jnp.logspace(0, 3.5, 80)

        predictions = []

        # LL predictions
        Cl_LL_vals = vmap(lambda ell: self.compute_Cl_LL_jax(Om, s8, ell))(ell_grid)

        for theta_val in self.theta_LL_plus:
            xi_LL_plus = self.hankel_j0(Cl_LL_vals, ell_grid, theta_val)
            predictions.append(xi_LL_plus)
        for theta_val in self.theta_LL_minus:
            xi_LL_minus = self.hankel_j4(Cl_LL_vals, ell_grid, theta_val)
            predictions.append(xi_LL_minus)

        # LE predictions
        for bin_idx in range(self.n_tomo_bins):
            z_edges = self.E_dist.limits
            z_gal = (z_edges[bin_idx] + z_edges[bin_idx + 1]) / 2.0

            Cl_LE_vals = vmap(lambda ell: self.compute_Cl_LE_jax(Om, s8, ell, z_gal))(ell_grid)

            for theta_val in self.theta_LE_plus[bin_idx]:
                xi_LE_plus = self.hankel_j0(Cl_LE_vals, ell_grid, theta_val)
                predictions.append(xi_LE_plus)

            for theta_val in self.theta_LE_minus[bin_idx]:
                xi_LE_minus = self.hankel_j4(Cl_LE_vals, ell_grid, theta_val)
                predictions.append(xi_LE_minus)

        # LP predictions
        for bin_idx in range(self.n_tomo_bins):
            z_edges = self.E_dist.limits
            z_gal = (z_edges[bin_idx] + z_edges[bin_idx + 1]) / 2.0

            Cl_LP_vals = vmap(lambda ell: self.compute_Cl_LP_jax(Om, s8, ell, z_gal))(ell_grid)

            for theta_val in self.theta_LP[bin_idx]:
                xi_LP = self.hankel_j2(Cl_LP_vals, ell_grid, theta_val)
                predictions.append(xi_LP)

        return jnp.array(predictions)

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

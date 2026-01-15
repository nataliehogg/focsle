"""
Fisher forecast computation module.

This module provides the main FisherForecast class for computing Fisher matrices
from cosmological observables.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union

# Enable float64 precision in JAX (must be set before importing jax.numpy)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacfwd

from .theory import TheoryJAX
from .data_loader import (
    load_covariance,
    build_full_covariance,
    detect_nbins,
    list_available_datasets,
    parse_dataset_name,
)



class FisherForecast:
    """
    Fisher matrix forecast calculator.

    This class provides a high-level interface for computing Fisher matrices
    for cosmological parameter constraints from LOS-LOS, LOS-shear, and
    LOS-position correlation functions.

    Args:
        data_dir: Path to the dataset directory containing covariance matrices
                  and distribution files
        lens_file: Path to lens catalog file (e.g., Euclid_lenses.txt)
        cosmo_fid: Dictionary of fiducial cosmology parameters (optional)
        verbose: Print progress messages

    Example:
        >>> from focsle import FisherForecast
        >>>
        >>> # Initialize with a specific dataset
        >>> forecast = FisherForecast(
        ...     data_dir='/path/to/data/Nlens=1e5_sigL=0.2_Nbin_z=6_...',
        ...     lens_file='/path/to/Euclid_lenses.txt'
        ... )
        >>>
        >>> # Setup the P(k) grid (expensive, do once)
        >>> forecast.setup(nOm=5, nAs=5)
        >>>
        >>> # Compute Fisher matrices
        >>> results = forecast.compute_fisher()
        >>>
        >>> # Save results
        >>> forecast.save_results('my_results.pkl')
    """

    def __init__(self, data_dir: str, lens_file: Optional[str] = None,
                 cosmo_fid: Optional[Dict] = None, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.lens_file = lens_file
        self.verbose = verbose

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Detect number of bins
        self.nbins = detect_nbins(self.data_dir)
        if verbose:
            print(f"Detected {self.nbins} tomographic bins from data")

        # Parse dataset info
        self.dataset_info = parse_dataset_name(self.data_dir.name)
        if verbose:
            print(f"Dataset parameters: {self.dataset_info}")

        # Initialize theory calculator
        self.theory = TheoryJAX(lens_file=lens_file, cosmo_fid=cosmo_fid)

        # Storage for results
        self.fisher_matrices = {}
        self.constraints = {}
        self.jacobian = None
        self.cov_blocks = None
        self.C_full = None
        self.C_inv = None
        self.sizes = None
        self._is_setup = False

    @classmethod
    def list_datasets(cls, data_root: str) -> List[str]:
        """
        List available datasets in a data directory.

        Args:
            data_root: Path to root data directory

        Returns:
            List of dataset directory names
        """
        return list_available_datasets(data_root)

    def setup(self, nOm: int = 5, nAs: int = 5, nz: int = 50, nk: int = 100,
              Om_range: Tuple[float, float] = (0.25, 0.40),
              As_range: Tuple[float, float] = (1.5e-9, 2.7e-9),
              theta_min_arcmin: Optional[float] = None):
        """
        Setup the forecast calculator.

        This performs the expensive CAMB P(k) grid computation and loads
        the galaxy distributions and angular bins.

        Args:
            nOm: Number of Omega_m grid points
            nAs: Number of A_s grid points
            nz: Number of redshift grid points
            nk: Number of wavenumber grid points
            Om_range: (min, max) range for Omega_m
            As_range: (min, max) range for A_s
            theta_min_arcmin: Minimum theta (arcmin); angular bins below are cut
        """
        if self.verbose:
            print("=" * 70)
            print("Setting up Fisher Forecast")
            print("=" * 70)

        # Setup P(k) grid
        if self.verbose:
            print("\nSetting up P(k) grid (this will take several minutes)...")
        self.theory.setup_Pk_grid(
            Om_range=Om_range, As_range=As_range,
            nOm=nOm, nAs=nAs, nz=nz, nk=nk,
            verbose=self.verbose
        )

        # Setup galaxy distributions
        if self.verbose:
            print("\nLoading galaxy distributions...")
        self.theory.setup_galaxy_distributions(str(self.data_dir), verbose=self.verbose)

        # Load angular bins
        if self.verbose:
            print("Loading angular bins...")
        self.theory.load_angular_bins(str(self.data_dir))
        self.theory.apply_theta_min_cut(theta_min_arcmin)
        if theta_min_arcmin is not None and self.verbose:
            print(f"Applying theta_min = {theta_min_arcmin} arcmin")
            for key, rep in self.theory.theta_cut_report.items():
                if rep['before'] != rep['after']:
                    print(f"  {key}: {rep['before']} -> {rep['after']} angles")

        # Load covariance matrices
        if self.verbose:
            print("\nLoading covariance matrices...")
        self.cov_blocks = load_covariance(
            str(self.data_dir), nbins=self.nbins, verbose=self.verbose
        )
        if theta_min_arcmin is not None:
            self._apply_theta_scale_cut()

        # Build full covariance
        if self.verbose:
            print("Building full covariance matrix...")
        self.C_full, self.sizes = build_full_covariance(self.cov_blocks)
        if self.verbose:
            print(f"  Shape: {self.C_full.shape}")
            print(f"  Data vector sizes: n_LL={self.sizes['n_LL']}, "
                  f"n_LE={self.sizes['n_LE']}, n_LP={self.sizes['n_LP']}")

        # Invert covariance
        if self.verbose:
            print("Inverting covariance matrix...")
        try:
            # Check condition number to warn about ill-conditioning
            cond_number = np.linalg.cond(self.C_full)
            if self.verbose:
                print(f"  Condition number: {cond_number:.2e}")
            if cond_number > 1e12:
                import warnings
                warnings.warn(
                    f"Covariance matrix is ill-conditioned (cond={cond_number:.2e}). "
                    "Results may have reduced numerical precision.",
                    RuntimeWarning
                )

            self.C_inv = self._robust_symmetric_inverse(self.C_full, verbose=self.verbose)
            if self.verbose:
                print("  Success!")
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular - cannot invert")

        self._is_setup = True
        if self.verbose:
            print("\nSetup complete!")
            if theta_min_arcmin is not None:
                print(f"Applied theta_min = {theta_min_arcmin} arcmin")

    def compute_fisher(self, param_names: List[str] = None) -> Dict:
        """
        Compute Fisher matrices for all probes.

        Args:
            param_names: Parameter names (default: ['Omega_m', 'sigma_8'])

        Returns:
            Dictionary containing Fisher matrices and constraints for each probe
        """
        if not self._is_setup:
            raise RuntimeError("Must call setup() before compute_fisher()")

        if param_names is None:
            param_names = ['Omega_m', 'sigma_8']

        if self.verbose:
            print("\n" + "=" * 70)
            print("Computing Fisher Matrices")
            print("=" * 70)

        # Get fiducial parameters
        fiducial = [self.theory.cosmo_fid['Omega_m'], self.theory.cosmo_fid['sigma8']]
        self.fiducial = fiducial

        if self.verbose:
            print(f"\nFiducial: Omega_m = {fiducial[0]:.4f}, sigma_8 = {fiducial[1]:.4f}")

        # Compute Jacobian
        if self.verbose:
            print("\nComputing Jacobian with JAX autodiff...")
            print("  (this may take a minute)")

        def data_vec_func(params):
            Om, s8 = params[0], params[1]
            return self.theory.predict_data_vector_jax(Om, s8)

        params_fid = jnp.array(fiducial)
        jacobian_func = jacfwd(data_vec_func)
        self.jacobian = jacobian_func(params_fid)

        if self.verbose:
            print(f"  Jacobian shape: {self.jacobian.shape}")
            print("  Done!")

        # Convert covariance inverse to JAX
        C_inv_jax = jnp.array(self.C_inv)

        # Extract sizes
        n_LL = self.sizes['n_LL']
        n_LE = self.sizes['n_LE']
        # n_LP = self.sizes['n_LP'] # not needed

        # Compute Fisher for each probe
        if self.verbose:
            print("\n" + "-" * 50)
            print("Computing Fisher matrices for individual probes")
            print("-" * 50)

        # LL only
        if self.verbose:
            print("\n1. LL only:")
        J_LL = self.jacobian[:n_LL, :]
        C_LL_inv = jnp.array(self._robust_symmetric_inverse(self.C_full[:n_LL, :n_LL]))
        F_LL = J_LL.T @ C_LL_inv @ J_LL
        self.fisher_matrices['LL'] = np.array(F_LL)
        self.constraints['LL'] = self._analyze_constraints(
            self.fisher_matrices['LL'], fiducial
        )
        self._print_constraints('LL')

        # LE only
        if self.verbose:
            print("\n2. LE only:")
        J_LE = self.jacobian[n_LL:n_LL + n_LE, :]
        C_LE_inv = jnp.array(self._robust_symmetric_inverse(
            self.C_full[n_LL:n_LL + n_LE, n_LL:n_LL + n_LE]
        ))
        F_LE = J_LE.T @ C_LE_inv @ J_LE
        self.fisher_matrices['LE'] = np.array(F_LE)
        self.constraints['LE'] = self._analyze_constraints(
            self.fisher_matrices['LE'], fiducial
        )
        self._print_constraints('LE')

        # LP only
        if self.verbose:
            print("\n3. LP only:")
        J_LP = self.jacobian[n_LL + n_LE:, :]
        C_LP_inv = jnp.array(self._robust_symmetric_inverse(
            self.C_full[n_LL + n_LE:, n_LL + n_LE:]
        ))
        F_LP = J_LP.T @ C_LP_inv @ J_LP
        self.fisher_matrices['LP'] = np.array(F_LP)
        self.constraints['LP'] = self._analyze_constraints(
            self.fisher_matrices['LP'], fiducial
        )
        self._print_constraints('LP')

        # Combined
        if self.verbose:
            print("\n4. Combined (LL + LE + LP):")
        F_combined = self.jacobian.T @ C_inv_jax @ self.jacobian
        self.fisher_matrices['Combined'] = np.array(F_combined)
        self.constraints['Combined'] = self._analyze_constraints(
            self.fisher_matrices['Combined'], fiducial
        )
        self._print_constraints('Combined')

        if self.verbose:
            print("\n" + "=" * 70)
            print("Fisher computation complete!")
            print("=" * 70)

        return {
            'fisher_matrices': self.fisher_matrices,
            'constraints': self.constraints,
            'fiducial': fiducial,
            'param_names': param_names,
        }

    def _robust_symmetric_inverse(self, M: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Compute a robust inverse of a symmetric matrix.

        For ill-conditioned matrices, the standard inverse can produce
        asymmetric results or matrices that are not positive definite.
        This method:
        1. Computes the inverse
        2. Symmetrizes to remove asymmetric numerical errors
        3. Ensures positive definiteness via eigenvalue clipping

        Args:
            M: Symmetric matrix to invert
            verbose: Print diagnostic information

        Returns:
            Robust symmetric positive-definite inverse
        """
        # Compute raw inverse and symmetrize
        M_inv_raw = np.linalg.inv(M)
        M_inv_sym = (M_inv_raw + M_inv_raw.T) / 2

        # Check if result is positive definite
        eigenvalues, eigenvectors = np.linalg.eigh(M_inv_sym)

        if np.any(eigenvalues <= 0):
            # Clip negative/zero eigenvalues to small positive value
            min_positive = np.max(eigenvalues) * 1e-10
            n_clipped = np.sum(eigenvalues <= 0)
            eigenvalues_clipped = np.maximum(eigenvalues, min_positive)

            if verbose:
                print(f"  Clipped {n_clipped} non-positive eigenvalues for numerical stability")

            # Reconstruct positive definite matrix
            M_inv_sym = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

        return M_inv_sym

    def _apply_theta_scale_cut(self):
        """Apply theta_min cut to covariance blocks to match theory predictions."""
        masks = self.theory.theta_masks

        def build_indices(mask_list):
            idx = []
            offset = 0
            for mask in mask_list:
                for keep in mask:
                    if keep:
                        idx.append(offset)
                    offset += 1
            return np.array(idx, dtype=int)

        # LL indices
        ll_masks = [masks['LL_plus'], masks['LL_minus']]
        idx_LL = build_indices(ll_masks)

        # LE indices (per bin: plus then minus)
        le_mask_list = []
        for i in range(self.theory.n_tomo_bins):
            le_mask_list.append(masks[f'LE_plus_{i}'])
            le_mask_list.append(masks[f'LE_minus_{i}'])
        idx_LE = build_indices(le_mask_list)

        # LP indices (per bin)
        lp_mask_list = [masks[f'LP_{i}'] for i in range(self.theory.n_tomo_bins)]
        idx_LP = build_indices(lp_mask_list)

        # Slice covariance blocks
        def slice_block(block, rows, cols):
            return block[np.ix_(rows, cols)]

        self.cov_blocks['LLLL'] = slice_block(self.cov_blocks['LLLL'], idx_LL, idx_LL)
        self.cov_blocks['LELE'] = slice_block(self.cov_blocks['LELE'], idx_LE, idx_LE)
        self.cov_blocks['LPLP'] = slice_block(self.cov_blocks['LPLP'], idx_LP, idx_LP)
        self.cov_blocks['LLLE'] = slice_block(self.cov_blocks['LLLE'], idx_LL, idx_LE)
        self.cov_blocks['LELP'] = slice_block(self.cov_blocks['LELP'], idx_LE, idx_LP)
        self.cov_blocks['LLLP'] = slice_block(self.cov_blocks['LLLP'], idx_LL, idx_LP)

    def _analyze_constraints(self, F: np.ndarray, fiducial: List[float]) -> Optional[Dict]:
        """Extract parameter constraints from Fisher matrix."""
        try:
            C = np.linalg.inv(F)
            errors = np.sqrt(np.diag(C))
            corr = C[0, 1] / (errors[0] * errors[1])

            return {
                'errors': errors,
                'covariance': C,
                'correlation': corr,
                'fractional_errors': errors / np.array(fiducial),
            }
        except np.linalg.LinAlgError:
            return None

    def _print_constraints(self, probe: str):
        """Print constraints for a probe."""
        if not self.verbose:
            return

        results = self.constraints.get(probe)
        if results:
            print(f"   sigma(Omega_m) = {results['errors'][0]:.4f} "
                  f"({100 * results['fractional_errors'][0]:.1f}%)")
            print(f"   sigma(sigma_8) = {results['errors'][1]:.4f} "
                  f"({100 * results['fractional_errors'][1]:.1f}%)")
            if probe == 'Combined':
                print(f"   Correlation: {results['correlation']:.3f}")
        else:
            print("   WARNING: Fisher matrix is singular!")

    def save_results(self, output_file: str):
        """
        Save results to a pickle file.

        Args:
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'fisher_matrices': self.fisher_matrices,
            'constraints': self.constraints,
            'fiducial': self.fiducial,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_dir': str(self.data_dir),
                'dataset_info': self.dataset_info,
                'n_LL': self.sizes['n_LL'],
                'n_LE': self.sizes['n_LE'],
                'n_LP': self.sizes['n_LP'],
                'nbins': self.nbins,
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results_dict, f)

        if self.verbose:
            print(f"\nResults saved to {output_path}")

    @staticmethod
    def load_results(results_file: str) -> Dict:
        """
        Load results from a pickle file.

        Args:
            results_file: Path to results file

        Returns:
            Dictionary containing Fisher results
        """
        with open(results_file, 'rb') as f:
            return pickle.load(f)

    def get_figure_of_merit(self, probe: str = 'Combined') -> float:
        """
        Compute the Figure of Merit (inverse area of error ellipse).

        Args:
            probe: Which probe to compute FoM for

        Returns:
            Figure of Merit value
        """
        F = self.fisher_matrices.get(probe)
        if F is None:
            raise ValueError(f"No Fisher matrix computed for probe: {probe}")

        return np.sqrt(np.linalg.det(F))

    def regularise_covariance(C, min_eigenvalue=1e-10):
        """Clip small/negative eigenvalues to ensure positive definiteness."""
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues_clipped = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

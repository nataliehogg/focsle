"""
Fisher forecast computation module.

This module provides the main FisherForecast class for computing Fisher matrices
from cosmological observables.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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
        self.C_LL_inv = None
        self.C_LE_inv = None
        self.C_LP_inv = None
        self.C_EE_inv = None
        self.C_EP_inv = None
        self.C_PP_inv = None
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
            size_str = f"  Data vector sizes: n_LL={self.sizes['n_LL']}, " \
                      f"n_LE={self.sizes['n_LE']}, n_LP={self.sizes['n_LP']}"
            if 'n_EE' in self.sizes:
                size_str += f", n_EE={self.sizes['n_EE']}, n_EP={self.sizes['n_EP']}, " \
                           f"n_PP={self.sizes['n_PP']}"
            print(size_str)

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

        # Pre-compute block inverses for Fisher matrix computation (avoids redundant inversions)
        if self.verbose:
            print("Pre-computing covariance block inverses...")
        n_LL = self.sizes['n_LL']
        n_LE = self.sizes['n_LE']
        n_LP = self.sizes['n_LP']

        self.C_LL_inv = self._robust_symmetric_inverse(
            self.C_full[:n_LL, :n_LL],
            verbose=False
        )
        self.C_LE_inv = self._robust_symmetric_inverse(
            self.C_full[n_LL:n_LL + n_LE, n_LL:n_LL + n_LE],
            verbose=False
        )
        self.C_LP_inv = self._robust_symmetric_inverse(
            self.C_full[n_LL + n_LE:n_LL + n_LE + n_LP, n_LL + n_LE:n_LL + n_LE + n_LP],
            verbose=False
        )

        # Check if new blocks exist (6-probe case)
        if 'n_EE' in self.sizes:
            n_EE = self.sizes['n_EE']
            n_EP = self.sizes['n_EP']
            n_PP = self.sizes['n_PP']

            # Compute EE block inverse
            start_EE = n_LL + n_LE + n_LP
            self.C_EE_inv = self._robust_symmetric_inverse(
                self.C_full[start_EE:start_EE + n_EE, start_EE:start_EE + n_EE],
                verbose=False
            )

            # Compute EP block inverse
            start_EP = start_EE + n_EE
            self.C_EP_inv = self._robust_symmetric_inverse(
                self.C_full[start_EP:start_EP + n_EP, start_EP:start_EP + n_EP],
                verbose=False
            )

            # Compute PP block inverse
            start_PP = start_EP + n_EP
            self.C_PP_inv = self._robust_symmetric_inverse(
                self.C_full[start_PP:, start_PP:],
                verbose=False
            )

            if self.verbose:
                print("  All block inverses cached (including EE, EP, PP)!")
        else:
            if self.verbose:
                print("  Block inverses cached!")

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
        n_LP = self.sizes['n_LP']

        # Compute Fisher for each probe
        if self.verbose:
            print("\n" + "-" * 50)
            print("Computing Fisher matrices for individual probes")
            print("-" * 50)

        # LL only (using pre-computed inverse)
        if self.verbose:
            print("\n1. LL only:")
        J_LL = self.jacobian[:n_LL, :]
        C_LL_inv = jnp.array(self.C_LL_inv)
        F_LL = J_LL.T @ C_LL_inv @ J_LL
        self.fisher_matrices['LL'] = np.array(F_LL)
        self.constraints['LL'] = self._analyze_constraints(
            self.fisher_matrices['LL'], fiducial
        )
        self._print_constraints('LL')

        # LE only (using pre-computed inverse)
        if self.verbose:
            print("\n2. LE only:")
        J_LE = self.jacobian[n_LL:n_LL + n_LE, :]
        C_LE_inv = jnp.array(self.C_LE_inv)
        F_LE = J_LE.T @ C_LE_inv @ J_LE
        self.fisher_matrices['LE'] = np.array(F_LE)
        self.constraints['LE'] = self._analyze_constraints(
            self.fisher_matrices['LE'], fiducial
        )
        self._print_constraints('LE')

        # LP only (using pre-computed inverse)
        if self.verbose:
            print("\n3. LP only:")
        J_LP = self.jacobian[n_LL + n_LE:n_LL + n_LE + n_LP, :]
        C_LP_inv = jnp.array(self.C_LP_inv)
        F_LP = J_LP.T @ C_LP_inv @ J_LP
        self.fisher_matrices['LP'] = np.array(F_LP)
        self.constraints['LP'] = self._analyze_constraints(
            self.fisher_matrices['LP'], fiducial
        )
        self._print_constraints('LP')

        # Check if new probes exist (6-probe case)
        if 'n_EE' in self.sizes:
            n_EE = self.sizes['n_EE']
            n_EP = self.sizes['n_EP']
            # n_PP = self.sizes['n_PP']

            # EE only
            if self.verbose:
                print("\n4. EE only:")
            start_EE = n_LL + n_LE + n_LP
            J_EE = self.jacobian[start_EE:start_EE + n_EE, :]
            C_EE_inv = jnp.array(self.C_EE_inv)
            F_EE = J_EE.T @ C_EE_inv @ J_EE
            self.fisher_matrices['EE'] = np.array(F_EE)
            self.constraints['EE'] = self._analyze_constraints(
                self.fisher_matrices['EE'], fiducial
            )
            self._print_constraints('EE')

            # EP only
            if self.verbose:
                print("\n5. EP only:")
            start_EP = start_EE + n_EE
            J_EP = self.jacobian[start_EP:start_EP + n_EP, :]
            C_EP_inv = jnp.array(self.C_EP_inv)
            F_EP = J_EP.T @ C_EP_inv @ J_EP
            self.fisher_matrices['EP'] = np.array(F_EP)
            self.constraints['EP'] = self._analyze_constraints(
                self.fisher_matrices['EP'], fiducial
            )
            self._print_constraints('EP')

            # PP only
            if self.verbose:
                print("\n6. PP only:")
            start_PP = start_EP + n_EP
            J_PP = self.jacobian[start_PP:, :]
            C_PP_inv = jnp.array(self.C_PP_inv)
            F_PP = J_PP.T @ C_PP_inv @ J_PP
            self.fisher_matrices['PP'] = np.array(F_PP)
            self.constraints['PP'] = self._analyze_constraints(
                self.fisher_matrices['PP'], fiducial
            )
            self._print_constraints('PP')

            # Combined (all 6 probes)
            if self.verbose:
                print("\n7. Combined (all 6 probes: LL + LE + LP + EE + EP + PP):")
        else:
            # Combined (3 probes only)
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

    def compute_custom_fisher(self, probe_names: List[str]) -> Dict:
        """
        Compute Fisher matrix for a custom combination of probes.

        Args:
            probe_names: List of probe names to combine, e.g. ['LL', 'EE', 'PP']

        Returns:
            Dictionary with Fisher matrix and constraints for the combination
        """
        if not self._is_setup:
            raise RuntimeError("Must call setup() and compute_fisher() first")

        # Build index mask for selected probes
        probe_indices = []
        n_LL = self.sizes['n_LL']
        n_LE = self.sizes['n_LE']
        n_LP = self.sizes['n_LP']

        offsets = {
            'LL': (0, n_LL),
            'LE': (n_LL, n_LL + n_LE),
            'LP': (n_LL + n_LE, n_LL + n_LE + n_LP),
        }

        # Add new probe offsets if they exist
        if 'n_EE' in self.sizes:
            n_EE = self.sizes['n_EE']
            n_EP = self.sizes['n_EP']
            n_PP = self.sizes['n_PP']
            offsets['EE'] = (n_LL + n_LE + n_LP, n_LL + n_LE + n_LP + n_EE)
            offsets['EP'] = (n_LL + n_LE + n_LP + n_EE, n_LL + n_LE + n_LP + n_EE + n_EP)
            offsets['PP'] = (n_LL + n_LE + n_LP + n_EE + n_EP, n_LL + n_LE + n_LP + n_EE + n_EP + n_PP)

        for probe in probe_names:
            if probe not in offsets:
                raise ValueError(f"Unknown probe: {probe}")
            start, end = offsets[probe]
            probe_indices.extend(range(start, end))

        probe_indices = np.array(probe_indices)

        # Extract Jacobian and covariance for selected probes
        J_custom = self.jacobian[probe_indices, :]
        C_custom = self.C_full[np.ix_(probe_indices, probe_indices)]
        C_custom_inv = self._robust_symmetric_inverse(C_custom, verbose=False)

        # Compute Fisher matrix
        F_custom = J_custom.T @ jnp.array(C_custom_inv) @ J_custom
        F_custom = np.array(F_custom)

        # Analyze constraints
        constraints = self._analyze_constraints(F_custom, self.fiducial)

        probe_label = '+'.join(probe_names)

        if self.verbose:
            print(f"\nCustom combination: {probe_label}")
            if constraints:
                print(f"   sigma(Omega_m) = {constraints['errors'][0]:.4f} "
                      f"({100 * constraints['fractional_errors'][0]:.1f}%)")
                print(f"   sigma(sigma_8) = {constraints['errors'][1]:.4f} "
                      f"({100 * constraints['fractional_errors'][1]:.1f}%)")

        return {
            'fisher_matrix': F_custom,
            'constraints': constraints,
            'fiducial': self.fiducial,
            'probe_combination': probe_label,
        }

    def _robust_symmetric_inverse(self, M: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Compute a robust inverse of a symmetric matrix.

        For ill-conditioned matrices, the standard inverse can produce
        asymmetric results or matrices that are not positive definite.
        This method:
        1. Ensures input matrix is positive definite (clips negative eigenvalues)
        2. Computes the inverse
        3. Symmetrizes to remove asymmetric numerical errors
        4. Ensures output is positive definite

        Args:
            M: Symmetric matrix to invert
            verbose: Print diagnostic information

        Returns:
            Robust symmetric positive-definite inverse
        """
        # First, ensure input matrix is positive definite
        M_sym = (M + M.T) / 2  # Symmetrize input
        eigenvalues_M, eigenvectors_M = np.linalg.eigh(M_sym)

        if np.any(eigenvalues_M <= 0):
            # Regularize input matrix by clipping negative eigenvalues
            min_positive = np.max(eigenvalues_M) * 1e-10
            n_clipped = np.sum(eigenvalues_M <= 0)
            eigenvalues_M_clipped = np.maximum(eigenvalues_M, min_positive)

            if verbose:
                print(f"  Regularized input matrix: clipped {n_clipped} non-positive eigenvalues")

            # Reconstruct positive definite input
            M_regularized = eigenvectors_M @ np.diag(eigenvalues_M_clipped) @ eigenvectors_M.T
        else:
            M_regularized = M_sym

        # Compute raw inverse and symmetrize
        M_inv_raw = np.linalg.inv(M_regularized)
        M_inv_sym = (M_inv_raw + M_inv_raw.T) / 2

        # Check if inverse is positive definite
        eigenvalues_inv, eigenvectors_inv = np.linalg.eigh(M_inv_sym)

        if np.any(eigenvalues_inv <= 0):
            # Clip negative/zero eigenvalues to small positive value
            min_positive = np.max(eigenvalues_inv) * 1e-10
            n_clipped = np.sum(eigenvalues_inv <= 0)
            eigenvalues_inv_clipped = np.maximum(eigenvalues_inv, min_positive)

            if verbose:
                print(f"  Clipped {n_clipped} non-positive eigenvalues in inverse for numerical stability")

            # Reconstruct positive definite inverse
            M_inv_sym = eigenvectors_inv @ np.diag(eigenvalues_inv_clipped) @ eigenvectors_inv.T

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

        # Handle new probe blocks if they exist
        if 'EEEE' in self.cov_blocks:
            # EE uses same mask structure as LE (plus/minus)
            idx_EE = idx_LE

            # EP and PP use same mask structure as LP
            idx_EP = idx_LP
            idx_PP = idx_LP

            self.cov_blocks['EEEE'] = slice_block(self.cov_blocks['EEEE'], idx_EE, idx_EE)
            self.cov_blocks['EPEP'] = slice_block(self.cov_blocks['EPEP'], idx_EP, idx_EP)
            self.cov_blocks['PPPP'] = slice_block(self.cov_blocks['PPPP'], idx_PP, idx_PP)
            self.cov_blocks['EEEP'] = slice_block(self.cov_blocks['EEEP'], idx_EE, idx_EP)
            self.cov_blocks['EEPP'] = slice_block(self.cov_blocks['EEPP'], idx_EE, idx_PP)
            self.cov_blocks['EPPP'] = slice_block(self.cov_blocks['EPPP'], idx_EP, idx_PP)

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

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'dataset_info': self.dataset_info,
            'n_LL': self.sizes['n_LL'],
            'n_LE': self.sizes['n_LE'],
            'n_LP': self.sizes['n_LP'],
            'nbins': self.nbins,
        }
        # Add new probe sizes if they exist
        if 'n_EE' in self.sizes:
            metadata['n_EE'] = self.sizes['n_EE']
            metadata['n_EP'] = self.sizes['n_EP']
            metadata['n_PP'] = self.sizes['n_PP']

        results_dict = {
            'fisher_matrices': self.fisher_matrices,
            'constraints': self.constraints,
            'fiducial': self.fiducial,
            'metadata': metadata
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results_dict, f)

        if self.verbose:
            print(f"\nResults saved to {output_path}")

    @staticmethod
    def load_results(results_file: str, results_dir: str = None) -> Dict:
        """
        Load results from a pickle file.

        Searches for the file in multiple locations:
        1. The exact path provided
        2. In results_dir (if provided)
        3. In standard locations: ./results, ../results, package_dir/../../results

        Args:
            results_file: Path to results file, or just the filename
            results_dir: Optional directory to search in

        Returns:
            Dictionary containing Fisher results
        """
        results_path = Path(results_file)

        # If it's already a valid file, use it directly
        if results_path.is_file():
            with open(results_path, 'rb') as f:
                return pickle.load(f)

        # Build list of directories to search
        search_dirs = []
        if results_dir:
            search_dirs.append(Path(results_dir))

        # Add standard locations
        search_dirs.extend([
            Path.cwd() / 'results',
            Path.cwd().parent / 'results',
            Path(__file__).parent.parent.parent / 'results',  # package_root/results
        ])

        # Search for the file
        filename = results_path.name
        for search_dir in search_dirs:
            candidate = search_dir / filename
            if candidate.is_file():
                with open(candidate, 'rb') as f:
                    return pickle.load(f)

        # If not found, raise helpful error
        searched = [str(d) for d in search_dirs if d.exists()]
        raise FileNotFoundError(
            f"Could not find '{filename}' in any of these locations:\n"
            + "\n".join(f"  - {d}" for d in searched)
            + f"\n\nAvailable files in results directories:\n"
            + "\n".join(f"  - {f.name}" for d in search_dirs if d.exists()
                       for f in d.glob("fisher_results_*.pkl"))
        )

    @staticmethod
    def list_saved_results(results_dir: str = None) -> List[str]:
        """
        List available saved Fisher results.

        Args:
            results_dir: Optional directory to search in

        Returns:
            List of available result filenames
        """
        search_dirs = []
        if results_dir:
            search_dirs.append(Path(results_dir))

        search_dirs.extend([
            Path.cwd() / 'results',
            Path.cwd().parent / 'results',
            Path(__file__).parent.parent.parent / 'results',
        ])

        results = []
        seen = set()
        for search_dir in search_dirs:
            if search_dir.exists():
                for f in sorted(search_dir.glob("fisher_results_*.pkl")):
                    if f.name not in seen:
                        results.append(f.name)
                        seen.add(f.name)
        return results

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

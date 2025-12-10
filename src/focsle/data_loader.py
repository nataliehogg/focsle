"""
Data loading utilities for focsle Fisher forecasting.

This module provides functions to load covariance matrices, redshift distributions,
and angular distributions from data directories.
"""

import sys
import types
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def _register_pickling_shims():
    """
    Register lightweight stand-ins for legacy classes referenced in collaborator pickles.

    The pickled files were created with classes in a separate `functions.*` package;
    we provide minimal shims here so unpickling works without that dependency.
    Pickle will overwrite these objects' __dict__ with stored state, so empty classes
    are sufficient.
    """
    class Redshift_Distributions:
        def __init__(self, *args, **kwargs):
            pass

    class Angular_Distributions:
        def __init__(self, *args, **kwargs):
            pass

    def redshift_distribution_Euclid(z):
        return 0.0 * z

    shim_functions = types.ModuleType('functions')
    shim_rd = types.ModuleType('functions.redshift_distributions')
    shim_ad = types.ModuleType('functions.angular_distributions')

    shim_rd.redshift_distribution_Euclid = redshift_distribution_Euclid
    shim_rd.Redshift_Distributions = Redshift_Distributions
    shim_ad.Angular_Distributions = Angular_Distributions
    shim_functions.redshift_distributions = shim_rd
    shim_functions.angular_distributions = shim_ad

    sys.modules.setdefault('functions', shim_functions)
    sys.modules.setdefault('functions.redshift_distributions', shim_rd)
    sys.modules.setdefault('functions.angular_distributions', shim_ad)


_register_pickling_shims()


def list_available_datasets(data_root: str) -> List[str]:
    """
    List all available dataset directories.

    Args:
        data_root: Path to the root data directory containing dataset folders

    Returns:
        List of dataset directory names
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    datasets = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and 'Nlens' in item.name:
            datasets.append(item.name)

    return datasets


def parse_dataset_name(dataset_name: str) -> Dict:
    """
    Parse dataset directory name to extract parameters.

    Args:
        dataset_name: Dataset directory name like 'Nlens=1e5_sigL=0.2_Nbin_z=6_...'

    Returns:
        Dictionary with parsed parameters
    """
    params = {}
    parts = dataset_name.split('_')

    for i, part in enumerate(parts):
        if '=' in part:
            key, val = part.split('=')
            # Handle special case of Nbin_z
            if key == 'Nbin' and i + 1 < len(parts) and parts[i + 1].startswith('z='):
                key = 'Nbin_z'
                val = parts[i + 1].split('=')[1]
            try:
                # Try to convert to number
                if 'e' in val.lower():
                    params[key] = float(val)
                elif '.' in val:
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = val

    return params


def detect_nbins(data_dir: str) -> int:
    """
    Detect number of tomographic bins from the data directory.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Number of tomographic bins
    """
    data_dir = Path(data_dir)

    # Try to detect from redshift distributions
    zdist_file = data_dir / 'redshift_distributions'
    if zdist_file.exists():
        with open(zdist_file, 'rb') as f:
            zdist = pickle.load(f)
        if 'E' in zdist and hasattr(zdist['E'], 'Nbinz'):
            return zdist['E'].Nbinz

    # Try to detect from angular distributions
    ang_file = data_dir / 'angular_distributions'
    if ang_file.exists():
        with open(ang_file, 'rb') as f:
            ang_dist = pickle.load(f)
        if 'LE_plus' in ang_dist:
            return len(ang_dist['LE_plus'])

    # Try to detect from covariance files
    cov_dir = data_dir / 'covariance' / 'LELE'
    if cov_dir.exists():
        # Count unique bin indices
        bins = set()
        for f in cov_dir.iterdir():
            if f.name.startswith('ccov_'):
                parts = f.name.replace('ccov_', '').split('_')
                if len(parts) >= 1:
                    bins.add(int(parts[0]))
        if bins:
            return max(bins) + 1

    # Fallback: try to parse from directory name
    params = parse_dataset_name(data_dir.name)
    if 'Nbin_z' in params:
        return params['Nbin_z']

    raise ValueError(f"Could not detect number of bins from {data_dir}")


def load_covariance_block(cov_type_dir: str, nbins: int) -> np.ndarray:
    """
    Load covariance block, handling both single-file and bin-by-bin structures.

    Args:
        cov_type_dir: Path to covariance type directory (e.g., LELE)
        nbins: Number of tomographic bins

    Returns:
        Combined covariance matrix (ccov + ncov + scov)
    """
    cov_type_dir = Path(cov_type_dir)

    # Try single-file format first (for backward compatibility)
    if (cov_type_dir / 'ccov').exists():
        with open(cov_type_dir / 'ccov', 'rb') as f:
            ccov = pickle.load(f)
        with open(cov_type_dir / 'ncov', 'rb') as f:
            ncov = pickle.load(f)
        with open(cov_type_dir / 'scov', 'rb') as f:
            scov = pickle.load(f)
        return ccov + ncov + scov

    # Otherwise, load bin-by-bin files and assemble
    cov_name = cov_type_dir.name

    if cov_name == 'LLLL':
        bins_list = [(0, 0)]
    elif cov_name in ['LELE', 'LPLP']:
        bins_list = [(i, j) for i in range(nbins) for j in range(nbins)]
    elif cov_name == 'LLLE':
        bins_list = [(0, j) for j in range(nbins)]
    elif cov_name == 'LELP':
        bins_list = [(i, j) for i in range(nbins) for j in range(nbins)]
    elif cov_name == 'LLLP':
        bins_list = [(0, j) for j in range(nbins)]
    else:
        raise ValueError(f"Unknown covariance type: {cov_name}")

    # Load all blocks
    blocks = {}
    for i, j in bins_list:
        if cov_name in ['LLLE', 'LLLP']:
            ccov_file = cov_type_dir / f'ccov_{j}'
            ncov_file = cov_type_dir / f'ncov_{j}'
            scov_file = cov_type_dir / f'scov_{j}'
        else:
            ccov_file = cov_type_dir / f'ccov_{i}_{j}'
            ncov_file = cov_type_dir / f'ncov_{i}_{j}'
            scov_file = cov_type_dir / f'scov_{i}_{j}'

        if not ccov_file.exists():
            ccov_file = cov_type_dir / 'ccov'
            ncov_file = cov_type_dir / 'ncov'
            scov_file = cov_type_dir / 'scov'

        with open(ccov_file, 'rb') as f:
            ccov = pickle.load(f)
        with open(ncov_file, 'rb') as f:
            ncov = pickle.load(f)
        with open(scov_file, 'rb') as f:
            scov = pickle.load(f)

        blocks[(i, j)] = ccov + ncov + scov

    # Assemble blocks
    if cov_name == 'LLLL':
        return blocks[(0, 0)]

    # Determine block sizes
    row_sizes = {}
    col_sizes = {}
    for (i, j), block in blocks.items():
        if i not in row_sizes:
            row_sizes[i] = block.shape[0]
        if j not in col_sizes:
            col_sizes[j] = block.shape[1]

    # Calculate total size
    if cov_name in ['LLLE', 'LLLP']:
        total_rows = row_sizes[0]
        total_cols = sum(col_sizes.values())
    else:
        total_rows = sum(row_sizes.values())
        total_cols = sum(col_sizes.values())

    cov_matrix = np.zeros((total_rows, total_cols))

    # Fill in blocks
    if cov_name in ['LLLE', 'LLLP']:
        col_offset = 0
        for j in range(nbins):
            if (0, j) in blocks:
                block = blocks[(0, j)]
                cov_matrix[:, col_offset:col_offset + block.shape[1]] = block
                col_offset += block.shape[1]
    else:
        row_offset = 0
        for i in range(nbins):
            col_offset = 0
            for j in range(nbins):
                if (i, j) in blocks:
                    block = blocks[(i, j)]
                    row_size = block.shape[0]
                    col_size = block.shape[1]
                    cov_matrix[row_offset:row_offset + row_size,
                               col_offset:col_offset + col_size] = block
                col_offset += col_sizes.get(j, 0)
            row_offset += row_sizes.get(i, 0)

    return cov_matrix


def load_covariance(data_dir: str, nbins: Optional[int] = None,
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Load all covariance matrices from a data directory.

    Args:
        data_dir: Path to dataset directory
        nbins: Number of tomographic bins (auto-detected if None)
        verbose: Print progress messages

    Returns:
        Dictionary mapping covariance type names to matrices
    """
    data_dir = Path(data_dir)
    cov_dir = data_dir / 'covariance'

    if nbins is None:
        nbins = detect_nbins(data_dir)
        if verbose:
            print(f"  Auto-detected {nbins} tomographic bins")

    cov = {}
    cov_types = ['LLLL', 'LELE', 'LPLP', 'LLLE', 'LELP', 'LLLP']

    for cov_type in cov_types:
        cov_type_dir = cov_dir / cov_type
        if not cov_type_dir.exists():
            if verbose:
                print(f"  Warning: {cov_type} directory not found, skipping...")
            continue

        if verbose:
            print(f"  Loading {cov_type}...")
        cov[cov_type] = load_covariance_block(cov_type_dir, nbins=nbins)
        if verbose:
            print(f"    Shape: {cov[cov_type].shape}")

    return cov


def build_full_covariance(cov_blocks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build full covariance matrix from blocks.

    Args:
        cov_blocks: Dictionary of covariance blocks

    Returns:
        Tuple of (full covariance matrix, dict of data vector sizes)
    """
    # Detect sizes from blocks
    n_LL = cov_blocks['LLLL'].shape[0]
    n_LE = cov_blocks['LELE'].shape[0]
    n_LP = cov_blocks['LPLP'].shape[0]

    sizes = {'n_LL': n_LL, 'n_LE': n_LE, 'n_LP': n_LP}
    n_total = n_LL + n_LE + n_LP

    C = np.zeros((n_total, n_total))

    # LL block
    C[:n_LL, :n_LL] = cov_blocks['LLLL']

    # LE block
    C[n_LL:n_LL + n_LE, n_LL:n_LL + n_LE] = cov_blocks['LELE']

    # LP block
    C[n_LL + n_LE:, n_LL + n_LE:] = cov_blocks['LPLP']

    # Cross-correlation blocks
    C[:n_LL, n_LL:n_LL + n_LE] = cov_blocks['LLLE']
    C[n_LL:n_LL + n_LE, :n_LL] = cov_blocks['LLLE'].T

    C[n_LL:n_LL + n_LE, n_LL + n_LE:] = cov_blocks['LELP']
    C[n_LL + n_LE:, n_LL:n_LL + n_LE] = cov_blocks['LELP'].T

    C[:n_LL, n_LL + n_LE:] = cov_blocks['LLLP']
    C[n_LL + n_LE:, :n_LL] = cov_blocks['LLLP'].T

    return C, sizes


def load_redshift_distributions(data_dir: str):
    """
    Load galaxy redshift distributions.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with 'E' and 'P' distribution objects
    """
    data_dir = Path(data_dir)
    zdist_file = data_dir / 'redshift_distributions'

    with open(zdist_file, 'rb') as f:
        zdist = pickle.load(f)

    return zdist


def load_angular_distributions(data_dir: str):
    """
    Load angular bin distributions.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with angular distribution objects
    """
    data_dir = Path(data_dir)
    ang_file = data_dir / 'angular_distributions'

    with open(ang_file, 'rb') as f:
        ang_dist = pickle.load(f)

    return ang_dist


def load_lens_catalog(lens_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load lens catalog from text file.

    Args:
        lens_file: Path to lens catalog file (e.g., Euclid_lenses.txt)

    Returns:
        Tuple of (lens redshifts, source redshifts)
    """
    lens_data = np.loadtxt(lens_file, comments='#')
    z_d_array = lens_data[:, 0]
    z_s_array = lens_data[:, 1]
    return z_d_array, z_s_array

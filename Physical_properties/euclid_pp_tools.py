# Copyright (c) Micol Bolzonella, Andrea Enia, Will Hartley, Giulia Papini, 2026.
#
# This file is subject to the terms and conditions defined in LICENSE.txt, which forms part of this source code
# package. No part of the package, including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in LICENSE.txt.


"""
euclid_pp_tools: galaxy physical properties at a chosen redshift from the Euclid DR1 PHZ output.

This module contains the functions used by the notebooks "Galaxy physical properties at a
chosen redshift from Euclid DR1" (the archive query functions are defined in the notebooks):

* table helpers (reading and writing catalogues, merging);
* access to the PHZ reference sample and to the realizations of the nearest neighbours;
* the main computation, `run_nnpz_evaluation`;
* diagnostic plots.

Version 1.2 (31 October 2026)
    Archive query functions moved back to the notebooks; reference sample read from the
    DR1 calibration volume (REFERENCE_DIR, REFERENCE_FILES); neighbour ids cast to integers
    in fix_nn_columns; percentages in the processing summary; new plot_zr_vs_zphot.
    All the functions previously defined in the notebooks moved here, with full docstrings.
Version 1.1 (October 12, 2026)
    Input validation, multiple redshifts per object kept in input order,
    `zr_extra_cols`, status 3 for unexpected errors.
"""

import multiprocessing
import time
import warnings
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import seaborn as sns
from astropy.table import Table, join
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# PHZ reference samples (realizations of the reference galaxies used by NNPZ), in the
# DR1 calibration volume. The realization files must be listed in order: the file numbers
# (pp_file, starting from 1) stored in the index refer to this order.
REFERENCE_DIR = '/data/euclid_dr1_cal/calproductsDR1/PHZ'
REFERENCE_FILES = {
    'DEEP': {'index': 'DEEP_PP_ref_11Aug2025_DR1_index.npy',
             'pp': ['galaxy_pp_ref_DR1_deep_1.npy', 'galaxy_pp_ref_DR1_deep_2.npy',
                    'galaxy_pp_ref_DR1_deep_3.npy', 'galaxy_pp_ref_DR1_deep_4.npy']},
    'WIDE': {'index': 'PP_RS_Index_10-2-0.npy',
             'pp': ['PP_RS_pp_data_1.10-2-0.npy', 'PP_RS_pp_data_2.10-2-0.npy']},
}

# Columns of the PHZ physical-parameters catalogue needed by run_nnpz_evaluation
NN_ID_COL, NN_WEIGHT_COL, NN_SCALE_COL = 'phz_pdf_nn_id', 'phz_pdf_nn_weight', 'phz_pdf_nn_scale'
REQUIRED_NN_COLS = [NN_ID_COL, NN_WEIGHT_COL, NN_SCALE_COL]

# Physical parameters available in the reference sample
PP_PARAMETERS = {'AV', 'MFUV', 'MNUV', 'MU', 'MB', 'MV', 'MR', 'MI', 'MZ', 'MVIS', 'MY',
                 'MJ', 'MH', 'MKs', 'REDSHIFT', 'SFHAGE', 'SFHTIMESCALE', 'SFR',
                 'STELLARMASS', 'MASSFORMED', 'STELLARMETALLICITY'}

# Parameters that depend on the flux scaling between target and neighbour:
# value_target = value_ref + factor * log10(scale)
PP_TOSCALE = {
    'STELLARMASS': 1, 'SFR': 1, 'MASSFORMED': 1,
    'MB': -2.5, 'MFUV': -2.5, 'MH': -2.5, 'MI': -2.5, 'MJ': -2.5, 'MKs': -2.5,
    'MNUV': -2.5, 'MR': -2.5, 'MU': -2.5, 'MV': -2.5, 'MVIS': -2.5, 'MY': -2.5, 'MZ': -2.5
}

# Processing status codes stored in the output column `wgt_warning`
STATUS_OK, STATUS_LOW_WEIGHT, STATUS_FAILURE, STATUS_ERROR = 0, 1, 2, 3
LOW_WEIGHT_THRESHOLD = 1e-30

_ID_ALIASES = ['object_id', 'id', 'source_id']

# Axis labels used in the plots
PARAM_LABELS = {'REDSHIFT': 'Redshift', 'STELLARMASS': r'$\log M_\star$', 'SFR': r'$\log {\rm SFR}$',
                'AV': r'$A_V$', 'SFHAGE': 'SFH age [Gyr]', 'MASSFORMED': r'$\log M_{\rm formed}$',
                'SFHTIMESCALE': r'SFH $\tau$ [Gyr]', 'STELLARMETALLICITY': r'$Z/Z_\odot$'}

# Global pointers for parallel processing worker access
data = None
index_df = None
pp = None
ID_colname = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_vector_string(vec_string, dtype=float):
    """
    Converts a string-serialized vector such as '(1.2, 3.4, ...)' into a numpy array.

    Parameters
    ----------
    vec_string : str or array-like
        The serialized vector. Inputs that are not strings are returned unchanged.
    dtype : type, optional
        `int` (or `np.int64`) for integer output, otherwise float (default is float).

    Returns
    -------
    numpy.ndarray or the unchanged input
    """
    if not isinstance(vec_string, str):
        return vec_string

    clean_str = vec_string.strip('()')
    parts = clean_str.split(',')
    arr = np.array(parts, dtype=float)

    if dtype == int or dtype == np.int64:
        return arr.astype(np.int64)
    return arr


def write_catalog(table, file_path, overwrite=True):
    """
    Writes an Astropy Table to disk as FITS or Parquet, depending on the file extension.

    Parameters
    ----------
    table : astropy.table.Table
        The table to write.
    file_path : str or pathlib.Path
        Output file. The format is chosen from the extension: '.fits' or '.fit' for FITS,
        '.parquet' for Parquet; any other extension raises a ValueError.
    overwrite : bool, optional
        Whether to overwrite an existing file (default is True).

    Returns
    -------
    None
    """
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()

    if ext == '.parquet':
        table.write(file_path, format='parquet', overwrite=overwrite)
    elif ext in ['.fits', '.fit']:
        table.write(file_path, format='fits', overwrite=overwrite)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Please use .fits or .parquet")

    print(f"✅ Successfully saved catalog to: {file_path}")


def read_catalog(file_path, memmap=True, verbose=False):
    """
    Reads a catalog from disk into an Astropy Table, automatically
    detecting if it is a FITS or Parquet file.

    Parameters
    ----------
    file_path : str
        Path to the file to read.
    memmap : bool, default True
        Whether to use memory mapping (highly recommended for FITS).
    verbose : bool, default False
        If True, prints detailed column metadata including data types and units.

    Returns
    -------
    astropy.table.Table
    """
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()

    if ext == '.parquet':
        table = Table.read(file_path, format='parquet')
    elif ext in ['.fits', '.fit']:
        table = Table.read(file_path, format='fits', memmap=memmap)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Please use .fits or .parquet")

    if verbose:
        print(f"File: {file_path}")
        print(f"Loaded {len(table)} rows and {len(table.colnames)} columns.")
        print("-" * 50)
        print("Column Definitions:")
        for col_name in table.colnames:
            col = table[col_name]
            info_parts = [f"name = '{col_name}'", f"dtype = '{col.dtype}'"]
            if len(col.shape) > 1:
                info_parts.append(f"dim = {col.shape[1:]}")
            if col.unit is not None:
                info_parts.append(f"unit = '{col.unit}'")
            print("    " + "; ".join(info_parts))
        print("-" * 50)

    return table


def first_entry_per_object(table, id_col='object_id'):
    """
    Returns a copy of `table` keeping only the first row of each `id_col`,
    preserving the original row order.

    Useful with the output of `run_nnpz_evaluation`, which can contain the same
    object at several redshifts: the first entry is the first redshift given
    for that object in the input redshift table.

    Parameters
    ----------
    table : astropy.table.Table
        Input table.
    id_col : str, optional
        ID column (default is 'object_id').

    Returns
    -------
    astropy.table.Table
    """
    _, idx = np.unique(np.asarray(table[id_col]), return_index=True)
    return table[np.sort(idx)]



# ---------------------------------------------------------------------------
# Archive and table helpers
# ---------------------------------------------------------------------------

def fix_nn_columns(table):
    """
    Makes sure that the nearest-neighbour columns are numeric (N, 30) arrays.

    Depending on the access route and on the file format, phz_pdf_nn_id,
    phz_pdf_nn_weight and phz_pdf_nn_scale can be returned as strings such as
    '(530951.0, ...)', and the neighbour ids as floats. String columns are converted,
    and float ids are cast to int64; the other numeric columns are left untouched.
    
    Parameters
    ----------
    table : astropy.table.Table
        Table with (some of) the nearest-neighbour columns. Modified in place.

    Returns
    -------
    astropy.table.Table
        The same table, with numeric nearest-neighbour columns.
    """
    for col, dtype in zip(REQUIRED_NN_COLS, (int, float, float)):
        if col not in table.colnames:
            print(f"{col}: not present")
        elif table[col].dtype.kind in 'US':
            table[col] = np.vstack([parse_vector_string(v, dtype=dtype) for v in table[col]])
            print(f"{col}: converted from string to {table[col].dtype}, shape {table[col].shape[1:]}")
        else:
            if dtype is int and table[col].dtype.kind == 'f':
                table[col] = np.ma.asarray(table[col]).astype(np.int64)
            print(f"{col}: numeric ({table[col].dtype}, shape {table[col].shape[1:]})")
    return table


def merge_catalogs(base_table, append_table):
    """
    Joins two catalogues on their ID columns.

    Columns present in both tables are taken from `append_table`. For IDs repeated
    in either table, only the first entry (in the original order) is kept.

    Parameters
    ----------
    base_table : astropy.table.Table
        Left table.
    append_table : astropy.table.Table or str
        Right table, or the path to a FITS/Parquet file.

    Returns
    -------
    astropy.table.Table
        Inner join of the two tables.
    """
    right_tab = read_catalog(append_table) if isinstance(append_table, str) else append_table.copy()
    left_tab = base_table.copy()

    left_id = _find_id_column(left_tab.colnames) or left_tab.colnames[0]
    right_id = _find_id_column(right_tab.colnames) or right_tab.colnames[0]

    cols_to_drop = list(set(left_tab.colnames).intersection(right_tab.colnames) - {left_id, right_id})
    if cols_to_drop:
        left_tab.remove_columns(cols_to_drop)

    for label, tab, id_col in (('left', left_tab, left_id), ('right', right_tab, right_id)):
        n_dup = len(tab) - len(np.unique(np.asarray(tab[id_col])))
        if n_dup > 0:
            print(f"⚠️ {n_dup} repeated IDs in the {label} table: only the first entry of each is used.")
    left_tab = first_entry_per_object(left_tab, left_id)
    right_tab = first_entry_per_object(right_tab, right_id)

    if left_id == right_id:
        return join(left_tab, right_tab, keys=left_id)
    return join(left_tab, right_tab, keys_left=left_id, keys_right=right_id)


# ---------------------------------------------------------------------------
# Reference sample and neighbour realizations
# ---------------------------------------------------------------------------

def _reference_file_paths(survey):
    """
    Paths of the index and realization files of a reference sample.

    Parameters
    ----------
    survey : {'DEEP', 'WIDE'}
        Reference sample.

    Returns
    -------
    key : str
        The survey, in upper case.
    index_path : pathlib.Path
        Index file.
    pp_paths : list of pathlib.Path
        Realization files, in the order of the file numbers stored in the index.

    Raises
    ------
    ValueError
        If the survey is not in REFERENCE_FILES.
    FileNotFoundError
        If any of the files is missing.
    """
    key = str(survey).upper().strip()
    if key not in REFERENCE_FILES:
        raise ValueError(f"Invalid survey '{survey}'. Must be one of {list(REFERENCE_FILES)}.")
    folder = Path(REFERENCE_DIR)
    index_path = folder / REFERENCE_FILES[key]['index']
    pp_paths = [folder / name for name in REFERENCE_FILES[key]['pp']]
    missing = [str(p) for p in [index_path] + pp_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Reference sample files not found: {missing}. "
                                "Check euclid_pp_tools.REFERENCE_DIR and REFERENCE_FILES.")
    return key, index_path, pp_paths


def load_reference_sample(euclid_survey='DEEP'):
    """
    Loads the index and memory-mapped realization files of the PHZ reference
    sample for the requested survey, using REFERENCE_DIR and REFERENCE_FILES.

    Parameters
    ----------
    euclid_survey : {'DEEP', 'WIDE'}, optional
        Reference sample to load (default is 'DEEP'). It must match the survey of
        the PHZ catalogue, whose neighbour ids refer to that reference sample.

    Returns
    -------
    index_df : pandas.DataFrame
        Index of the reference sample (indexed by reference id), giving the file
        number (`pp_file`) and row (`pp_offset`) where each reference galaxy is stored.
    pp : list of numpy structured arrays (memory-mapped)
        Physical-parameter realizations of the reference galaxies.
    """
    survey_key, index_path, pp_paths = _reference_file_paths(euclid_survey)
    print(f"Loading {survey_key} reference sample from {index_path.parent}...")
    index_df = pd.DataFrame(np.load(index_path)).set_index('id')
    pp = [np.load(p, allow_pickle=True, mmap_mode='r') for p in pp_paths]
    return index_df, pp

def get_nn_realizations(galaxy_row, index_df, pp):
    """
    Collects the realizations of all the nearest neighbours of one object.

    Parameters
    ----------
    galaxy_row : astropy.table.Row (or dict-like)
        Row containing phz_pdf_nn_id, phz_pdf_nn_weight and phz_pdf_nn_scale.
    index_df, pp : outputs of `load_reference_sample`.

    Returns
    -------
    posterior : numpy structured array
        All realizations of all neighbours found in the reference sample
        (typically 30 neighbours x 50 realizations), with the reference values
        (i.e. NOT yet rescaled to the target flux).
    weights : numpy array
        NNPZ weight of the neighbour each realization belongs to.
    scales : numpy array
        NNPZ flux scale of the neighbour each realization belongs to.
    Returns (None, None, None) if no neighbour is found in the reference sample.
    """
    kNN_IDs = parse_vector_string(galaxy_row[NN_ID_COL], dtype=int)
    kNN_wgt = parse_vector_string(galaxy_row[NN_WEIGHT_COL], dtype=float)
    kNN_scal = parse_vector_string(galaxy_row[NN_SCALE_COL], dtype=float)

    # Masked entries (if any) are neutralised: unknown id, zero weight, unit scale
    kNN_IDs = np.ma.filled(np.ma.asarray(kNN_IDs), -1)
    kNN_wgt = np.ma.filled(np.ma.asarray(kNN_wgt), 0.0)
    kNN_scal = np.ma.filled(np.ma.asarray(kNN_scal), 1.0)

    all_realizations, all_wgts, all_scales = [], [], []
    for nn_id, wgt, scal in zip(kNN_IDs, kNN_wgt, kNN_scal):
        try:
            ref_info = index_df.loc[nn_id]
        except KeyError:
            continue
        sub = pp[int(ref_info['pp_file']) - 1][int(ref_info['pp_offset'])]
        all_realizations.append(sub)
        all_wgts.append(np.full(len(sub), wgt))
        all_scales.append(np.full(len(sub), scal))

    if not all_realizations:
        return None, None, None
    return (np.concatenate(all_realizations), np.concatenate(all_wgts),
            np.concatenate(all_scales))


def apply_scaling(values, param, scales):
    """
    Rescales the reference values of a parameter to the flux of the target.

    Parameters
    ----------
    values : numpy.ndarray
        Values of `param` for the reference realizations.
    param : str
        Parameter name (e.g. 'STELLARMASS'). Parameters not in PP_TOSCALE are not changed.
    scales : numpy.ndarray
        NNPZ flux scale of the neighbour each realization belongs to.

    Returns
    -------
    numpy.ndarray
        value + factor * log10(scale), with factor = +1 for log masses and SFR and
        -2.5 for absolute magnitudes.
    """
    if param in PP_TOSCALE:
        return values + PP_TOSCALE[param] * np.log10(scales)
    return values


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def fast_weighted_quantile(values, weights, quantiles):
    """
    Weighted quantiles, from the interpolation of the weighted cumulative distribution.

    Parameters
    ----------
    values : numpy.ndarray
        Sample values.
    weights : numpy.ndarray
        Non-negative weights, same length as `values`.
    quantiles : list of float
        Quantiles to compute, between 0 and 1.

    Returns
    -------
    numpy.ndarray
        The weighted quantiles, in the order of `quantiles`.
    """
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def extract_quantiles(i_target, quantiles, params, zr_col, zr_min_num, zr_tol):
    """
    Evaluates the PPs of one (object, redshift) pair. Internal worker of run_nnpz_evaluation.

    It reads the module globals `data`, `index_df`, `pp` and `ID_colname`, set by
    run_nnpz_evaluation before starting the parallel processes.

    Parameters
    ----------
    i_target : int
        Row of `data` to process.
    quantiles, params, zr_col, zr_min_num, zr_tol
        As in run_nnpz_evaluation.

    Returns
    -------
    list
        [id, quantiles/avg/bestnn for each parameter, NN_number, NN_valid, zr_window, status].
    """
    global data, index_df, pp, ID_colname
    galaxy_row = data[i_target]
    target_id = galaxy_row[ID_colname]
    n_values = len(params) * (len(quantiles) + 2)

    try:
        posterior, weights, scalings = get_nn_realizations(galaxy_row, index_df, pp)
        if posterior is None:
            return [target_id] + [-99.0] * n_values + [0, 0, 0.0, STATUS_FAILURE]

        z_target = galaxy_row[zr_col]
        delta_z = np.abs(posterior['REDSHIFT'] - z_target)

        # Adaptive window: grow by 0.01 until zr_min_num weighted realizations or zr_tol
        valid_mask = weights > 0
        z_window = 0.01
        mask = (delta_z < z_window) & valid_mask
        while (np.sum(mask) < zr_min_num) and (z_window < zr_tol):
            z_window += 0.01
            mask = (delta_z < z_window) & valid_mask

        nn_number = np.sum(delta_z < z_window)
        nn_valid = np.sum(mask)

        sub_wgt = weights[mask]
        max_wgt = np.max(sub_wgt) if len(sub_wgt) > 0 else 0

        if nn_valid == 0 or max_wgt <= 0:
            return ([target_id] + [-99.0] * n_values
                    + [int(nn_number), int(nn_valid), float(z_window), STATUS_FAILURE])

        wgt_warning = STATUS_LOW_WEIGHT if (max_wgt < LOW_WEIGHT_THRESHOLD) else STATUS_OK
        sub_wgt = sub_wgt / max_wgt

        sub_post = posterior[mask]
        sub_scal = scalings[mask]
        best_idx = np.argmax(sub_wgt)

        row_results = [target_id]
        for p in params:
            vals = apply_scaling(sub_post[p], p, sub_scal)
            row_results.extend(fast_weighted_quantile(vals, sub_wgt, quantiles).tolist())
            row_results.append(np.average(vals, weights=sub_wgt))
            row_results.append(vals[best_idx])

        row_results.extend([int(nn_number), int(nn_valid), float(z_window), wgt_warning])
        return row_results

    except Exception:
        # Unexpected problem for this object only (e.g. corrupted entry): flagged separately
        return [target_id] + [-99.0] * n_values + [0, 0, 0.0, STATUS_ERROR]


def _find_id_column(colnames):
    """Returns the first column name matching a known ID alias, or None."""
    return next((c for c in colnames if c.lower() in _ID_ALIASES), None)


def _quantile_label(q):
    """Column label of a quantile: 'median' for 0.5, otherwise the percentile (e.g. '16')."""
    return 'median' if np.isclose(q, 0.5) else f"{int(round(q * 100))}"


def run_nnpz_evaluation(nnpz_table, zr_table, euclid_survey='DEEP',
                        out_file='PHZ_PP_results.parquet', params='all',
                        quantiles=[0.16, 0.5, 0.84], zr_col='z_rederived',
                        zr_min_num=5, zr_tol=0.3, cores=0, zr_extra_cols=None):
    """
    Evaluate physical-parameter posteriors from the Euclid NNPZ output at a
    redshift chosen by the user ("rederived" redshift, zr).

    For each (object, redshift) pair, the realizations of the object's nearest
    neighbours in the DR1 reference sample are collected; only those whose
    redshift lies within a window around zr are kept. The half-width of the
    window starts at 0.01 and grows by 0.01 until at least `zr_min_num`
    realizations with positive weight are found, or until it reaches `zr_tol`.
    Weighted quantiles, weighted average and best-neighbour value are then
    computed for each requested parameter.

    Parameters
    ----------
    nnpz_table : str or astropy.table.Table
        PHZ physical-parameters catalogue (path to a FITS/Parquet file, or an
        in-memory Table). REQUIRED columns: an ID column (object_id),
        phz_pdf_nn_id, phz_pdf_nn_weight, phz_pdf_nn_scale. Other columns are ignored.
    zr_table : str or astropy.table.Table
        Catalogue with the redshifts at which the PPs must be evaluated
        (spectroscopic, machine learning, user-defined, ...). REQUIRED columns:
        an ID column matching the one in `nnpz_table`, and `zr_col`.
        The same object can appear several times with different redshifts:
        all of them are processed and returned (see Notes).
    euclid_survey : {'DEEP', 'WIDE'}, default 'DEEP'
        Must match the survey of `nnpz_table`: the neighbour ids refer to the
        reference sample of that survey.
    out_file : str or None, default 'PHZ_PP_results.parquet'
        Output file. The format is chosen from the extension: '.fits' or '.fit' for FITS,
        '.parquet' for Parquet. If None, nothing is written to disk.
    params : 'all' or list of str, default 'all'
        Physical parameters to evaluate (see Notes).
    quantiles : list of float, default [0.16, 0.5, 0.84]
        Quantiles of the weighted posterior to compute. 0.5 is labelled 'median'.
    zr_col : str, default 'z_rederived'
        Name of the redshift column in `zr_table`.
    zr_min_num : int, default 5
        Minimum number of realizations with weight > 0 required in the window.
    zr_tol : float, default 0.3
        Maximum half-width (Δz) of the adaptive redshift window.
    cores : int, default 0
        Number of CPU cores. If 0, uses all available cores minus one.
    zr_extra_cols : list of str or None, default None
        Columns of `zr_table` to copy into the output (e.g. ['spe_rank'] or a
        column identifying the source of each redshift).

    Returns
    -------
    out_table : astropy.table.Table
        One row per (object, redshift) pair with z > 0 found in both catalogues,
        in the same order as in `zr_table`. Also written to `out_file` if given.

    Output columns
    --------------
    <ID>                       object identifier (as in `nnpz_table`)
    <zr_col>                   redshift at which the PPs were evaluated
    <zr_extra_cols>            copied from `zr_table`, if requested
    phz_pp_zr_<q>_<param>      weighted quantile q of the parameter (e.g. _16_, _median_, _84_)
    phz_pp_zr_avg_<param>      weighted average
    phz_pp_zr_bestnn_<param>   value of the highest-weight realization in the window
    NN_number                  number of realizations within the final window (any weight)
    NN_valid                   number of realizations within the final window with weight > 0
                               (note: realizations, not neighbours; ~50 per neighbour)
    zr_window                  final half-width of the redshift window
    wgt_warning                processing status:
                                 0 = clean;
                                 1 = low weight: the largest weight in the window is below
                                     LOW_WEIGHT_THRESHOLD (1e-30), i.e. the neighbours compatible
                                     with zr are poor matches to the photometry;
                                 2 = failure: no realization with weight > 0 within `zr_tol`
                                     (typically zr far from the redshifts of the neighbours,
                                     e.g. photo-z outliers); values set to -99;
                                 3 = unexpected error while processing the object; values set to -99.
    Masses and SFRs are in log10 units and are rescaled to the target flux with
    the NNPZ scale (absolute magnitudes likewise).

    Notes
    -----
    Multiple redshifts per object: rows with identical (ID, redshift) are processed
    once; different redshifts for the same ID are all evaluated, so IDs can repeat
    in the output. The output keeps the input order, hence the first entry of an ID
    is the first redshift given for it (see `first_entry_per_object`).

    Available parameters (definitions and units in the Euclid DR1 Data Model):
    ===================  =======================================================
    Galaxy properties    'REDSHIFT', 'STELLARMASS', 'MASSFORMED', 'SFR',
                         'SFHAGE', 'SFHTIMESCALE', 'STELLARMETALLICITY', 'AV'
    Rest-frame absolute  'MFUV', 'MNUV', 'MU', 'MB', 'MV', 'MR', 'MI', 'MZ',
    magnitudes           'MVIS', 'MY', 'MJ', 'MH', 'MKs'
    ===================  =======================================================
    """
    global data, index_df, pp, ID_colname
    start_time = time.time()

    # 1. PHZ catalogue
    if isinstance(nnpz_table, Table):
        print("Using in-memory Astropy Table for the PHZ catalogue.")
        data_tab = nnpz_table
    else:
        print(f"Reading PHZ catalogue from file: {nnpz_table}")
        data_tab = read_catalog(nnpz_table)
    print(f"Number of PHZ entries: {len(data_tab)}")

    ID_colname = _find_id_column(data_tab.colnames) or data_tab.colnames[0]
    missing = [c for c in REQUIRED_NN_COLS if c not in data_tab.colnames]
    if missing:
        raise KeyError(f"The PHZ catalogue lacks the required column(s) {missing}. "
                       f"Required: an ID column plus {REQUIRED_NN_COLS}.")

    # 2. Redshift catalogue
    if isinstance(zr_table, Table):
        print("Using in-memory Astropy Table for the redshift catalogue.")
        ext_zr_table = zr_table.copy()
    else:
        print(f"Reading redshift catalogue from file: {zr_table}")
        ext_zr_table = read_catalog(zr_table)
    print(f"Number of redshift entries: {len(ext_zr_table)}")

    if zr_col not in ext_zr_table.colnames:
        raise KeyError(f"Redshift column '{zr_col}' not found in the redshift catalogue.")
    zr_extra_cols = list(zr_extra_cols) if zr_extra_cols else []
    missing = [c for c in zr_extra_cols if c not in ext_zr_table.colnames]
    if missing:
        raise KeyError(f"zr_extra_cols {missing} not found in the redshift catalogue.")

    if ID_colname not in ext_zr_table.colnames:
        matched_id = _find_id_column(ext_zr_table.colnames)
        if not matched_id:
            raise KeyError(f"Could not find an ID column matching '{ID_colname}' in the redshift catalogue.")
        print(f"Warning: using '{matched_id}' as ID column of the redshift catalogue.")
        ext_zr_table.rename_column(matched_id, ID_colname)

    keep_cols = [ID_colname, zr_col] + [c for c in zr_extra_cols if c not in (ID_colname, zr_col)]
    ext_zr_table = ext_zr_table[keep_cols]
    ext_zr_table['_zr_order'] = np.arange(len(ext_zr_table))

    # Remove exact duplicates of (ID, redshift), keeping the first occurrence
    key_df = pd.DataFrame({'id': np.asarray(ext_zr_table[ID_colname]),
                           'z': np.ma.filled(np.ma.asarray(ext_zr_table[zr_col], dtype=float), np.nan)})
    dup = key_df.duplicated().values
    if dup.any():
        ext_zr_table = ext_zr_table[~dup]
        print(f"Removed {dup.sum()} duplicated (ID, redshift) entries from the redshift catalogue.")

    _, id_counts = np.unique(np.asarray(ext_zr_table[ID_colname]), return_counts=True)
    multi_z_count = np.sum(id_counts > 1)
    if multi_z_count > 0:
        print(f"ℹ️ {multi_z_count} objects have more than one redshift: "
              "the PPs will be evaluated at all of them.")

    # 3. Match the two catalogues (only the columns needed by the computation)
    data = join(data_tab[[ID_colname] + REQUIRED_NN_COLS], ext_zr_table,
                keys=ID_colname, join_type='inner')
    zvals = np.ma.filled(np.ma.asarray(data[zr_col], dtype=float), np.nan)
    data = data[zvals > 0]
    data.sort('_zr_order')  # restore the order of the redshift catalogue
    tot_rows = len(data)
    print(f"Running analysis on {tot_rows} matched (object, redshift) pairs with z > 0.")
    if tot_rows == 0:
        raise ValueError("No matched entries with z > 0: check the ID columns and the redshift column.")

    # 4. Reference sample and parameter validation
    index_df, pp = load_reference_sample(euclid_survey=euclid_survey)
    available = PP_PARAMETERS & set(pp[0].dtype.names)
    if params == 'all' or params == ['all']:
        params_to_process = sorted(available)
    else:
        params_to_process = list(params)
        unknown = [p for p in params_to_process if p not in available]
        if unknown:
            raise ValueError(f"Unknown parameter(s) {unknown}. Available: {sorted(available)}")

    q_labels = [_quantile_label(q) for q in quantiles]
    if len(set(q_labels)) != len(q_labels):
        raise ValueError(f"Quantiles {quantiles} produce duplicated column labels {q_labels}.")

    # 5. Parallel processing
    num_workers = cores if cores > 0 else max(1, multiprocessing.cpu_count() - 1)
    print(f"Processing on {num_workers} cores...\n")
    worker_func = partial(extract_quantiles, quantiles=quantiles, params=params_to_process,
                          zr_col=zr_col, zr_min_num=zr_min_num, zr_tol=zr_tol)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(worker_func, range(tot_rows), chunksize=256),
                            total=tot_rows, desc='Evaluating quantiles',
                            bar_format='{l_bar}{bar}| Elapsed: {elapsed} ETA: {remaining}'))

    # 6. Formatting
    print('\nFormatting results...')
    cols = [ID_colname]
    for p in params_to_process:
        name = p.lower()
        cols.extend([f"phz_pp_zr_{lab}_{name}" for lab in q_labels])
        cols.extend([f"phz_pp_zr_avg_{name}", f"phz_pp_zr_bestnn_{name}"])
    cols.extend(['NN_number', 'NN_valid', 'zr_window', 'wgt_warning'])
    final_df = pd.DataFrame(results, columns=cols)

    if not np.array_equal(np.asarray(data[ID_colname]), final_df[ID_colname].values):
        raise RuntimeError("ID mismatch between input and results: please report this issue.")

    stat_cols = [c for c in cols if c not in [ID_colname, 'NN_number', 'NN_valid', 'wgt_warning']]
    final_df[stat_cols] = final_df[stat_cols].apply(pd.to_numeric).astype(np.float32)
    final_df['NN_number'] = final_df['NN_number'].astype(np.int32)
    final_df['NN_valid'] = final_df['NN_valid'].astype(np.int32)
    final_df['wgt_warning'] = final_df['wgt_warning'].astype(np.int16)

    out_table = Table.from_pandas(final_df)
    out_table[ID_colname] = data[ID_colname]
    # redshift (and extra columns) right after the ID
    for k, c in enumerate([zr_col] + [c for c in zr_extra_cols if c not in (ID_colname, zr_col)]):
        out_table.add_column(data[c], name=c, index=1 + k)

    out_table.meta['EXTNAME'] = 'PHZ_PP_ZR'
    out_table.meta['SURVEY'] = str(euclid_survey).upper()
    out_table.meta['ZR_COL'] = zr_col
    out_table.meta['ZR_TOL'] = zr_tol
    out_table.meta['ZR_MIN'] = zr_min_num
    out_table.meta['QUANTILS'] = str(quantiles)
    out_table.meta['PARAMS'] = 'ALL' if params_to_process == sorted(available) else ','.join(params_to_process)

    # 7. Summary (percentages with respect to the (object, z) pairs processed)
    status = final_df['wgt_warning'].values
    n_tot = len(status)
    ok = status <= STATUS_LOW_WEIGHT

    def _line(label, n):
        return f"{label:<30s}{n:>7d} ({100. * n / n_tot:5.1f}%)"

    print("\n" + "=" * 48)
    print("               PROCESSING SUMMARY")
    print("=" * 48)
    print(f"{'(object, z) pairs processed:':<30s}{n_tot:>7d}")
    print(f"{'Distinct objects:':<30s}{len(np.unique(final_df[ID_colname])):>7d}")
    print(_line("🟩 Clean (status 0):", np.sum(status == STATUS_OK)))
    print(_line("⚠️ Low weight (status 1):", np.sum(status == STATUS_LOW_WEIGHT)))
    print(_line("🔴 Failures, -99 (status 2):", np.sum(status == STATUS_FAILURE)))
    if np.any(status == STATUS_ERROR):
        print(_line("❌ Unexpected errors (status 3):", np.sum(status == STATUS_ERROR)))
    if ok.any():
        print(f"{'Mean NN_valid (status 0-1):':<30s}{final_df['NN_valid'][ok].mean():>7.1f}")
    print("=" * 48)

    if out_file:
        write_catalog(out_table, out_file)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    return out_table


# ---------------------------------------------------------------------------
# Diagnostics and plots
# ---------------------------------------------------------------------------

def plot_weight_distribution(table, threshold=LOW_WEIGHT_THRESHOLD):
    """
    Prints statistics of the NNPZ weights and plots their distribution.

    Parameters
    ----------
    table : astropy.table.Table
        Table containing the phz_pdf_nn_weight column.
    threshold : float, optional
        Low-weight threshold drawn on the plot (default is LOW_WEIGHT_THRESHOLD, 1e-30).

    Returns
    -------
    None
    """
    weights = np.ma.filled(np.ma.asarray(table[NN_WEIGHT_COL], dtype=float), np.nan).ravel()
    weights = weights[np.isfinite(weights)]
    positive = weights[weights > 0]
    n_zero = np.sum(weights == 0)

    print(f"Total weights         : {len(weights)}")
    print(f"Weights = 0           : {n_zero}")
    print(f"Minimum positive (> 0): {positive.min() if len(positive) else 'none'}")
    print(f"Maximum weight        : {weights.max()}")
    print(f"0 < weight < {threshold:.0e}  : {np.sum((weights > 0) & (weights < threshold))}")
    if len(positive) == 0:
        return

    plt.figure(figsize=(9, 5))
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 50)
    plt.hist(positive, bins=bins, color='royalblue', alpha=0.7, edgecolor='k', label='Positive weights')
    plt.axvline(threshold, color='red', ls='--', label=f'Low-weight threshold ({threshold:.0e})')
    plt.xscale('log')
    plt.xlabel(NN_WEIGHT_COL)
    plt.ylabel('N')
    plt.title(f"Distribution of NNPZ weights ({n_zero} exact zeros excluded)")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.show()


def pick_example(catalog, status, min_valid=0, id_col='object_id'):
    """
    Picks an example object with a given processing status.

    Parameters
    ----------
    catalog : astropy.table.Table
        Output of run_nnpz_evaluation (or a table merged with it), with the columns
        wgt_warning and NN_valid.
    status : int
        Value of wgt_warning to select (0, 1, 2 or 3).
    min_valid : int, optional
        Objects with NN_valid >= min_valid are preferred, if any (default is 0).
    id_col : str, optional
        ID column (default is 'object_id').

    Returns
    -------
    int or None
        The ID of the first matching object, or None if there is none.
    """
    sel = catalog[catalog['wgt_warning'] == status]
    if len(sel) == 0:
        return None
    rich = sel[sel['NN_valid'] >= min_valid]
    return (rich if len(rich) else sel)[id_col][0]


STATUS_STYLES = {STATUS_OK: ('o', 'tab:green', 'clean (0)'),
                 STATUS_LOW_WEIGHT: ('^', 'tab:orange', 'low weight (1)'),
                 STATUS_FAILURE: ('x', 'tab:red', 'failure (2)'),
                 STATUS_ERROR: ('s', 'k', 'error (3)')}


def plot_zr_vs_zphot(zr_results, pp_table, zr_col='spe_z', zp_col='phz_pp_median_redshift',
                     id_col='object_id', out_limit=0.15, zmax=None, figsize=(6, 6)):
    """
    Plots the new redshifts zr against the photometric redshifts of the PHZ catalogue,
    with one symbol per processing status (wgt_warning).

    All the (object, redshift) pairs of the output of run_nnpz_evaluation are shown,
    so objects with several redshifts appear several times.

    Parameters
    ----------
    zr_results : astropy.table.Table
        Output of run_nnpz_evaluation, with `id_col`, `zr_col` and wgt_warning.
    pp_table : astropy.table.Table
        PHZ catalogue with `id_col` and `zp_col`.
    zr_col : str, optional
        Column of the new redshifts (default is 'spe_z').
    zp_col : str, optional
        Column of the photometric redshifts (default is 'phz_pp_median_redshift').
    id_col : str, optional
        ID column (default is 'object_id').
    out_limit : float, optional
        |zr - zphot| / (1 + zr) drawn as dotted lines (default is 0.15).
    zmax : float, optional
        Upper limit of both axes (default: from the data).
    figsize : tuple, optional
        Figure size (default (6, 6)).

    Returns
    -------
    None
    """
    zp = pp_table[[id_col, zp_col]]
    t = join(zr_results[[id_col, zr_col, 'wgt_warning']], first_entry_per_object(zp, id_col),
             keys=id_col, join_type='inner')
    zr, zph, status = (np.asarray(t[zr_col], float), np.asarray(t[zp_col], float),
                       np.asarray(t['wgt_warning']))
    good = np.isfinite(zr) & np.isfinite(zph) & (zph > -1)
    zr, zph, status = zr[good], zph[good], status[good]
    if zmax is None:
        zmax = 1.05 * max(zr.max(), zph.max()) if len(zr) else 1.

    fig, ax = plt.subplots(figsize=figsize)
    for st, (marker, color, label) in STATUS_STYLES.items():
        sel = status == st
        if sel.any():
            ax.scatter(zr[sel], zph[sel], marker=marker, s=18, color=color, alpha=0.7,
                       label=f"{label}: {sel.sum()} ({100. * sel.sum() / len(status):.1f}%)")
    z_line = np.linspace(0, zmax, 200)
    ax.plot(z_line, z_line, 'k-', lw=1)
    ax.plot(z_line, z_line + out_limit * (1 + z_line), 'k:', lw=1)
    ax.plot(z_line, z_line - out_limit * (1 + z_line), 'k:', lw=1)
    ax.set_xlim(0, zmax)
    ax.set_ylim(0, zmax)
    ax.set_xlabel(f'$z_r$ ({zr_col})')
    ax.set_ylabel(rf'$z_{{\rm phot}}$ ({zp_col})')
    ax.set_title(f'{len(zr)} (object, redshift) pairs')
    ax.legend(title='wgt_warning', loc='upper left')
    ax.grid(alpha=0.3)
    plt.show()


def plot_neighbor_redshifts(target_id, catalog, index_df, pp, zr_col='spe_z',
                            zp_col='phz_pp_median_redshift', id_col='object_id'):
    """
    Plots the redshifts of the realizations of the 30 nearest neighbours of one object.

    The plot shows all the realizations (unweighted and weighted), the redshift zr at
    which the PPs were evaluated, the original photometric redshift, and the window
    around zr finally adopted by run_nnpz_evaluation.

    Parameters
    ----------
    target_id : int
        ID of the object to plot.
    catalog : astropy.table.Table
        Table with the nearest-neighbour columns, `zr_col`, zr_window and wgt_warning
        (e.g. the PHZ catalogue merged with the output of run_nnpz_evaluation).
    index_df, pp
        Outputs of load_reference_sample.
    zr_col : str, optional
        Column with the redshift used in the computation (default is 'spe_z').
    zp_col : str, optional
        Column with the original photometric redshift; not drawn if absent
        (default is 'phz_pp_median_redshift').
    id_col : str, optional
        ID column (default is 'object_id').

    Returns
    -------
    None
    """
    sel = catalog[catalog[id_col] == target_id]
    if len(sel) == 0:
        print(f"ID {target_id} not found in the catalogue.")
        return
    row = sel[0]
    if row['wgt_warning'] >= STATUS_FAILURE:
        print(f"⚠️ Object {target_id} has wgt_warning = {row['wgt_warning']}: "
              "no valid realization within the maximum window.")

    posterior, weights, _ = get_nn_realizations(row, index_df, pp)
    if posterior is None:
        print(f"No neighbour of ID {target_id} found in the reference sample.")
        return
    z_real = posterior['REDSHIFT']
    z_r, z_window = row[zr_col], row['zr_window']

    plt.figure(figsize=(9, 5))
    plt.hist(z_real, bins=50, color='lightgray', alpha=0.7, label='All realizations', density=True)
    if np.sum(weights) > 0:
        plt.hist(z_real, bins=50, weights=weights, color='royalblue', histtype='step', linewidth=2,
                 label='Weighted realizations', density=True)
    plt.axvspan(z_r - z_window, z_r + z_window, color='green', alpha=0.15,
                label=rf'Adopted window ($\pm {z_window:.2f}$)')
    plt.axvline(z_r, color='red', ls='--', lw=2, label=rf'$z_r$ = {z_r:.3f}')
    if zp_col in catalog.colnames and row[zp_col] != -99:
        plt.axvline(row[zp_col], color='blue', ls=':', lw=2, label=rf'$z_{{\rm phot}}$ = {row[zp_col]:.3f}')
    plt.title(f"Redshifts of the neighbours: ID {target_id} (wgt_warning = {row['wgt_warning']})")
    plt.xlabel('Redshift')
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()


def _plot_corner_cell(ax, x, y, weights, levels, color, cmap, old_x, new_x, old_y=None, new_y=None,
                      xlabel='', ylabel='', diagonal=False, bins=12):
    """
    Draws one panel of a corner plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Panel to draw on.
    x, y : numpy.ndarray
        Values on the x and y axes (`y` is ignored on the diagonal).
    weights : numpy.ndarray
        Weights of the realizations.
    levels : list of float
        Density levels of the contours.
    color : color
        Colour of the points and 1D histograms.
    cmap : str
        Colour map of the 2D histograms.
    old_x, new_x, old_y, new_y : float
        Catalogue and new medians to mark (NaN values are not drawn).
    xlabel, ylabel : str, optional
        Axis labels.
    diagonal : bool, optional
        If True, draws a 1D weighted histogram of `x` (default is False).
    bins : int, optional
        Number of histogram bins (default is 12).

    Returns
    -------
    None
    """
    if diagonal:
        sns.histplot(x=x, weights=weights, bins=bins, color=color, ax=ax, element='step')
        ax.set_yticks([])
        ax.set_ylabel('')
        if np.isfinite(old_x):
            ax.axvline(old_x, color='red', ls='--', lw=2, zorder=10)
        if np.isfinite(new_x):
            ax.axvline(new_x, color='lime', ls='-', lw=2, zorder=10)
    else:
        sns.scatterplot(x=x, y=y, size=weights, sizes=(5, 45), color=color, alpha=0.5, legend=False, ax=ax)
        sns.histplot(x=x, y=y, weights=weights, bins=bins, pthresh=min(levels), cmap=cmap, cbar=False, ax=ax)
        try:
            sns.kdeplot(x=x, y=y, weights=weights, levels=levels, color='k', linewidths=1, ax=ax)
        except Exception:
            pass  # too few points or degenerate distribution: no contours
        if np.isfinite(old_x) and np.isfinite(old_y):
            ax.plot(old_x, old_y, marker='s', color='red', ms=8, mec='k', ls='None', zorder=10)
        if np.isfinite(new_x) and np.isfinite(new_y):
            ax.plot(new_x, new_y, marker='*', color='lime', ms=14, mec='k', ls='None', zorder=10)
        ax.set_ylabel(ylabel, size=14)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlabel, size=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', labelsize=12)


def _plot_dynamic_corner(title, variables, labels, old_medians, new_medians, weights,
                         levels, cmap='flare'):
    """
    Draws a corner plot for any number of variables.

    Parameters
    ----------
    title : str
        Figure title.
    variables : list of numpy.ndarray
        One array of realization values per parameter.
    labels : list of str
        Axis label of each parameter.
    old_medians, new_medians : list of float
        Catalogue and new medians of each parameter (NaN values are not drawn).
    weights : numpy.ndarray
        Weights of the realizations.
    levels : list of float
        Density levels of the contours.
    cmap : str, optional
        Colour map (default is 'flare').

    Returns
    -------
    None
    """
    n = len(variables)
    color = sns.color_palette(cmap, as_cmap=True)(0.1)
    fig, axes = plt.subplots(n, n, figsize=(max(8, 4 * n),) * 2, squeeze=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.axis('off')
                continue
            diag = (r == c)
            _plot_corner_cell(ax, variables[c], None if diag else variables[r], weights, levels, color, cmap,
                              old_medians[c], new_medians[c],
                              None if diag else old_medians[r], None if diag else new_medians[r],
                              xlabel=labels[c] if r == n - 1 else '',
                              ylabel=labels[r] if (c == 0 and not diag) else '',
                              diagonal=diag)
            if r < n - 1:
                ax.set_xticklabels([])
            if c > 0 and not diag:
                ax.set_yticklabels([])
    handles = [mlines.Line2D([], [], color='red', marker='s', ls='None', ms=8, mec='k',
                             label='PHZ catalogue median'),
               mlines.Line2D([], [], color='lime', marker='*', ls='None', ms=14, mec='k',
                             label=r'Median at $z_r$')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.90, 0.90), fontsize=14)
    plt.suptitle(title, size=18, y=0.92)
    plt.show()


def corner_for_object(obj_id, catalog, params, index_df, pp, zr_col='spe_z', window_only=False,
                      id_col='object_id', levels=(np.exp(-4.5), np.exp(-2), np.exp(-0.5), 0.8), cmap='flare'):
    """
    Corner plot of the PPs of one object, from the realizations of its nearest neighbours.

    The catalogue median (red square) is the redshift-marginalised value published in the
    PHZ catalogue; the median at zr (green star) is the one computed by run_nnpz_evaluation.

    Parameters
    ----------
    obj_id : int
        ID of the object to plot.
    catalog : astropy.table.Table
        Table with the nearest-neighbour columns, `zr_col` and zr_window; the columns
        phz_pp_median_<param> and phz_pp_zr_median_<param> are used for the markers if present.
    params : list of str
        Reference-sample parameters to plot, e.g. ['REDSHIFT', 'STELLARMASS', 'SFR'].
    index_df, pp
        Outputs of load_reference_sample.
    zr_col : str, optional
        Column with the redshift used in the computation (default is 'spe_z').
    window_only : bool, optional
        If False (default), shows all the realizations; if True, only those within the
        adopted window around zr with positive weight, i.e. those used for the new values.
    id_col : str, optional
        ID column (default is 'object_id').
    levels : tuple of float, optional
        Density levels of the contours.
    cmap : str, optional
        Colour map (default is 'flare').

    Returns
    -------
    None
    """
    sel = catalog[catalog[id_col] == obj_id]
    if len(sel) == 0:
        print(f"ID {obj_id} not found in the catalogue.")
        return
    row = sel[0]
    posterior, weights, scales = get_nn_realizations(row, index_df, pp)
    if posterior is None:
        print(f"No neighbour of ID {obj_id} found in the reference sample.")
        return

    if window_only:
        mask = (np.abs(posterior['REDSHIFT'] - row[zr_col]) < row['zr_window']) & (weights > 0)
        if mask.sum() < 2:
            print(f"Only {mask.sum()} realizations in the window: nothing to plot.")
            return
        posterior, weights, scales = posterior[mask], weights[mask], scales[mask]

    def _value(col):
        return float(row[col]) if col in catalog.colnames and row[col] != -99 else np.nan

    variables = [apply_scaling(posterior[p], p, scales) for p in params]
    old_medians = [_value(f'phz_pp_median_{p.lower()}') for p in params]
    new_medians = [_value(f'phz_pp_zr_median_{p.lower()}') for p in params]
    labels = [PARAM_LABELS.get(p, p) for p in params]
    mode = 'realizations in the window' if window_only else 'all realizations'
    title = f"ID {obj_id}, $z_r$ = {row[zr_col]:.3f} ({mode}, {len(weights)} realizations)"
    _plot_dynamic_corner(title, variables, labels, old_medians, new_medians,
                         weights / np.max(weights), list(levels), cmap)


def stats_zszp(zs, zp, out_limit=0.15, verbose=True):
    """
    Statistics of photometric versus spectroscopic redshifts.

    Parameters
    ----------
    zs, zp : array-like
        Spectroscopic and photometric redshifts.
    out_limit : float, optional
        Outlier threshold on |zs - zp| / (1 + zs) (default is 0.15).
    verbose : bool, optional
        If True (default), prints the statistics.

    Returns
    -------
    tuple of float
        (NMAD, sigma, bias, outlier percentage); sigma and bias exclude the outliers.
    """
    zs, zp = np.asarray(zs), np.asarray(zp)
    dz = (zs - zp) / (1. + zs)
    outliers = np.abs(dz) > out_limit
    eta = outliers.mean() * 100.
    nmad = 1.4826 * np.median(np.abs(dz))
    bias = np.mean(dz[~outliers])
    sigma = np.std(dz[~outliers])
    if verbose:
        print(f"bias = {bias:.4f}  NMAD = {nmad:.4f}  sigma = {sigma:.4f}  outliers = {eta:.2f}%")
    return nmad, sigma, bias, eta


def scatter_zszp_1plusz(zs, zp, title, out_limit=0.15, zmin=0.0, zmax=6.0, bins=100, dens_thresh=2,
                        use_density=True, cmap='viridis', point_color='tab:blue', point_size=1,
                        point_alpha=0.3, figsize=(5, 5), dpi=150):
    """
    Plots photometric versus spectroscopic redshifts on log(1+z) axes, with statistics.

    Parameters
    ----------
    zs, zp : array-like
        Spectroscopic and photometric redshifts.
    title : str
        Plot title.
    out_limit : float, optional
        Outlier threshold on |dz|/(1+z), drawn as dotted lines (default is 0.15).
    zmin, zmax : float, optional
        Axis limits (default 0 and 6).
    bins : int, optional
        Number of bins of the 2D histogram (default is 100).
    dens_thresh : int, optional
        Cells with fewer objects are drawn as individual points (default is 2).
    use_density : bool, optional
        True (default) for a 2D histogram, False for a scatter plot.
    cmap : str, optional
        Colour map of the 2D histogram (default is 'viridis').
    point_color, point_size, point_alpha : optional
        Style of the scatter plot.
    figsize : tuple, optional
        Figure size (default (5, 5)).
    dpi : int, optional
        Figure resolution (default 150).

    Returns
    -------
    None
    """
    zs, zp = np.asarray(zs), np.asarray(zp)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    ax.set_xscale('function', functions=(np.log1p, np.expm1))
    ax.set_yscale('function', functions=(np.log1p, np.expm1))
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(zmin, zmax)

    if use_density:
        edges = np.expm1(np.linspace(np.log1p(max(zmin, 1e-3)), np.log1p(zmax), bins + 1))
        counts, xedges, yedges = np.histogram2d(zs, zp, bins=[edges, edges])
        ax.pcolormesh(xedges, yedges, np.ma.masked_less(counts, dens_thresh).T, cmap=cmap, norm=LogNorm())
        ix = np.clip(np.searchsorted(xedges, zs, side='right') - 1, 0, len(xedges) - 2)
        iy = np.clip(np.searchsorted(yedges, zp, side='right') - 1, 0, len(yedges) - 2)
        sparse = counts[ix, iy] < dens_thresh
        ax.plot(zs[sparse], zp[sparse], '.', color='DarkGray', markersize=point_size * 2)
    else:
        ax.plot(zs, zp, '.', color=point_color, markersize=point_size, alpha=point_alpha)

    z_line = np.linspace(zmin, zmax, 300)
    ax.plot([zmin, zmax], [zmin, zmax], 'r--', lw=1)
    ax.plot(z_line, z_line + out_limit * (1 + z_line), 'r:', lw=1.5)
    ax.plot(z_line, z_line - out_limit * (1 + z_line), 'r:', lw=1.5)

    nmad, sigma, bias, eta = stats_zszp(zs, zp, out_limit)
    text = (f"$N = {len(zs)}$\n$\\langle\\Delta z/(1+z)\\rangle = {bias:.4f}$\n"
            f"NMAD $= {nmad:.4f}$\n$\\eta = {eta:.2f}\\%$")
    ax.text(0.05, 0.95, text, fontsize=8, transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))

    major = np.array([0, 0.5, 1, 2, 3, 4, 5, 6])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(major[major <= zmax]))
        axis.set_minor_locator(FixedLocator(np.arange(0, zmax + 0.1, 0.1)))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.set_xlabel(r'$z_{\rm spec}$')
    ax.set_ylabel(r'$z_{\rm phot}$')
    ax.set_title(title, fontsize=10)
    plt.show()


def diagnostic_check_all_params(survey='DEEP'):
    """
    Prints the minimum and maximum of every quantity in the reference-sample files.

    It reads the whole files, so it can take a while.

    Parameters
    ----------
    survey : {'DEEP', 'WIDE'}, optional
        Reference sample to check (default is 'DEEP').

    Returns
    -------
    None
    """
    _, _, pp_paths = _reference_file_paths(survey)
    for p in pp_paths:
        ref = np.load(p, allow_pickle=True, mmap_mode='r')
        print(f"\n{p.name} ({len(ref)} galaxies)")
        print(f"{'Column':<22} | {'Min':>12} | {'Max':>12}")
        for col in ref.dtype.names:
            vals = np.asarray(ref[col]).ravel()
            print(f"{col:<22} | {np.min(vals):12.4f} | {np.max(vals):12.4f}")

### version 1.1 of euclid_pp_tools.py
### October 12, 2026

import os, multiprocessing, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from astropy.table import Table, join
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Location of the PHZ reference samples (physical-parameter realizations of
# the reference galaxies used by NNPZ). Can be overridden by the user, e.g.
#   import euclid_pp_tools; euclid_pp_tools.REFERENCE_PATHS['DEEP'] = '/my/path/'
REFERENCE_PATHS = {
    'DEEP': '/media/team_workspaces/Euclid-Consortium/data/PHZ/DR1_reference_samples/DEEP_PP/',
    'WIDE': '/media/team_workspaces/Euclid-Consortium/data/PHZ/DR1_reference_samples/WIDE_PP/'
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
    Cleans string-serialized vectors like '(1.2, 3.4, ...)' into numpy arrays.
    Handles 'float-as-string' parsing issues seamlessly. Non-string inputs
    are returned unchanged.
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
    Writes an Astropy Table to disk, automatically selecting the correct
    engine (FITS or Parquet) based on the file extension.
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
    """
    _, idx = np.unique(np.asarray(table[id_col]), return_index=True)
    return table[np.sort(idx)]


# ---------------------------------------------------------------------------
# Reference sample and neighbour realizations
# ---------------------------------------------------------------------------

def load_reference_sample(euclid_survey='DEEP'):
    """
    Loads the index and memory-mapped realization files of the PHZ reference
    sample for the requested survey ('DEEP' or 'WIDE'), using REFERENCE_PATHS.

    Returns
    -------
    index_df : pandas.DataFrame
        Index of the reference sample (indexed by reference id), giving the file
        number (`pp_file`) and row (`pp_offset`) where each reference galaxy is stored.
    pp : list of numpy structured arrays (memory-mapped)
        Physical-parameter realizations of the reference galaxies.
    """
    survey_key = str(euclid_survey).upper().strip()
    if survey_key not in REFERENCE_PATHS:
        raise ValueError(f"Invalid euclid_survey '{euclid_survey}'. Must be 'DEEP' or 'WIDE'.")

    ppref_path = Path(REFERENCE_PATHS[survey_key])
    if not ppref_path.is_dir():
        raise FileNotFoundError(f"Reference sample folder not found: {ppref_path}. "
                                "Check euclid_pp_tools.REFERENCE_PATHS.")
    print(f"Loading {survey_key} reference sample from memory-mapped files...")

    index_df = pd.DataFrame(np.load(next(ppref_path.glob("*index*")))).set_index('id')

    # Natural sorting (1, 2, ..., 10) of the realization files
    raw_paths = list(ppref_path.glob('galaxy_pp_ref*.npy'))
    pp_paths = sorted(raw_paths, key=lambda p: int(p.stem.split('_')[-1]))
    pp = [np.load(str(p), allow_pickle=True, mmap_mode='r') for p in pp_paths]

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
    """Rescales reference values of `param` to the target flux (no-op for unscaled parameters)."""
    if param in PP_TOSCALE:
        return values + PP_TOSCALE[param] * np.log10(scales)
    return values


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def fast_weighted_quantile(values, weights, quantiles):
    """Fast weighted quantile calculation via numpy interpolation of the weighted CDF."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def extract_quantiles(i_target, quantiles, params, zr_col, zr_min_num, zr_tol):
    """Worker: evaluates the posterior of one (object, redshift) pair. Uses module globals."""
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
    return next((c for c in colnames if c.lower() in _ID_ALIASES), None)


def _quantile_label(q):
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
        Output file (.fits or .parquet). If None, nothing is written to disk.
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

    # 7. Summary
    status = final_df['wgt_warning'].values
    ok = status <= STATUS_LOW_WEIGHT
    print("\n" + "=" * 46)
    print("              PROCESSING SUMMARY")
    print("=" * 46)
    print(f"(object, z) pairs processed:  {len(final_df)}")
    print(f"Distinct objects:             {len(np.unique(final_df[ID_colname]))}")
    print(f"🟩 Clean (status 0):          {np.sum(status == STATUS_OK)}")
    print(f"⚠️ Low weight (status 1):      {np.sum(status == STATUS_LOW_WEIGHT)}")
    print(f"🔴 Failures, -99 (status 2):   {np.sum(status == STATUS_FAILURE)}")
    if np.any(status == STATUS_ERROR):
        print(f"❌ Unexpected errors (status 3): {np.sum(status == STATUS_ERROR)}")
    if ok.any():
        print(f"Mean NN_valid (status 0-1):   {final_df['NN_valid'][ok].mean():.1f}")
    print("=" * 46)

    if out_file:
        write_catalog(out_table, out_file)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    return out_table

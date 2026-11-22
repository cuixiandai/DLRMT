import csv
import numpy as np
import pandas as pd

_NAN_TOKENS = ('NAN', 'NaN', 'nan', 'NA', 'nd', 'ND', '')
_ENCODING   = 'utf-8-sig'

def load_data(name, start=None, end=None):
########################################################################################
    if name == 'BaffleCreek':
        path = 'Datasets/BaffleCreek/baffle-creek-buoy-quality-2014.csv'
        cols = ['Temp_degC', 'pH', 'Turbidity_NTU', 'Chloraphylla_ugL', 'DO_mg']
        return _load_columns(path, cols, skiprows=1) 
########################################################################################
    if name == 'Burnett':
        path = 'Datasets/Burnett/burnett-river-2019.csv'
        cols = ['Temp degC', 'pH', 'Turbidity NTU', 'Chloraphylla ugL', 'DO mg']
        return _load_columns(path, cols, skiprows=2)
########################################################################################
    if name == 'Guadalupe':
        path = 'Datasets/Guadalupe/sonde_surface_bottom_combined.csv'
        cols = ['Sal', 'DO', 'pH', 'Temp']
        return _load_columns(path, cols, skiprows=1)
########################################################################################
    raise ValueError(f'Unknown dataset name: {name}')

def _read_header(path, encoding=_ENCODING):
    with open(path, 'r', encoding=encoding, newline='') as f:
        return next(csv.reader(f))


def _normalize_to_float(data_str):
    """String array (N, C) -> float32; unify handling of missing tokens and double quotes."""
    if data_str.size == 0:
        return data_str.astype(np.float32)

    # Strip leading/trailing whitespace and double quotes in one pass: ' "NAN"' -> 'NAN', '"7.6"' -> '7.6'
    cleaned = np.empty(data_str.shape, dtype='U32')
    cleaned[:] = np.char.strip(data_str, ' "')

    mask = np.isin(cleaned, _NAN_TOKENS)
    cleaned[mask] = 'nan'
    return cleaned.astype(np.float32)


def _read_fallback_csv(path, indices, skiprows, encoding):
    """Fallback when np.loadtxt fails: pad each row to the required column count, then take the target columns."""
    ncol = max(indices) + 1
    rows = []
    with open(path, 'r', encoding=encoding, newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            if i < skiprows:
                continue
            if len(row) < ncol:
                row = row + [''] * (ncol - len(row))
            rows.append([row[idx] for idx in indices])

    if not rows:
        return np.empty((0, len(indices)), dtype='U32')
    return np.array(rows, dtype='U32')


def _load_columns(path, col_names, skiprows, encoding=_ENCODING):
    header = _read_header(path, encoding)
    try:
        indices = [header.index(name) for name in col_names]
    except ValueError as e:
        raise KeyError(
            f'"{path}" 找不到列 {col_names}，实际 header={header}'
        ) from e

    try:
        with open(path, 'r', encoding=encoding, newline='') as f:
            data = np.loadtxt(
                f,
                delimiter=',',
                dtype=str,
                skiprows=skiprows,
                usecols=indices,
                ndmin=2,
            )
    except ValueError:
        data = _read_fallback_csv(path, indices, skiprows, encoding)

    return _normalize_to_float(data)

if __name__ == '__main__':
    for name in ('BaffleCreek', 'Burnett'):
        # For BaffleCreek, reproduce range filtering with config's data_start/data_end (verify change 1)
        if name == 'BaffleCreek':
            from config import get_config as _gc
            _cfg = _gc(name)
            T = load_data(name, _cfg.get('data_start'), _cfg.get('data_end'))
        else:
            T = load_data(name)
        print(f'=== {name} ===')
        print('shape :', T.shape)
        print('dtype :', T.dtype)
        print('NaN   :', np.isnan(T).sum(axis=0))
        print('head  :\n', T[:3])
        print()
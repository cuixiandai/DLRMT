import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def fill_missing(T, fill_boundary=True, max_fill_len=48):
    """
    Forward-fill an (N, C) time series to preserve temporal continuity.

    fill_boundary=True (default, backward compatible):
        Extrapolate the leading NaN segment with the first valid value and the
        trailing NaN segment with the last valid value; the result contains no
        NaNs (unless a column is entirely NaN).
    fill_boundary=False (recommended):
        Fill only interior NaNs between the first and last valid values; keep
        leading/trailing NaNs to avoid anomalous extrapolation from the training
        tail contaminating the start of validation/test sets.
    max_fill_len=48:
        Limit the maximum number of steps in a single continuous forward fill;
        keep NaNs beyond the threshold to avoid turning a corrupted test segment
        into a constant (which would make the model learn constant outputs and
        cause R2/KGE to become nan).
    """
    T = T.astype(np.float32).copy()
    N, C = T.shape
    for c in range(C):
        col = T[:, c]
        idx = np.where(~np.isnan(col))[0]
        if len(idx) == 0:
            continue
        first, last = idx[0], idx[-1]

        # Forward-fill interior NaNs, but cap the maximum continuous fill length
        last_valid_idx = first
        for i in range(first + 1, last + 1):
            if np.isnan(col[i]):
                if i - last_valid_idx <= max_fill_len:
                    col[i] = col[last_valid_idx]
                else:
                    # Missing run is too long: do not fill, keep NaN
                    pass
            else:
                last_valid_idx = i

        if fill_boundary:
            # Extrapolate at the boundaries
            col[:first] = col[first]
            col[last + 1:] = col[last]
        # When fill_boundary=False, leading/trailing NaNs are preserved
        T[:, c] = col
    return T


def winsorize(T, window=48, n_sigma=3.0):
    """
    Rolling-window winsorization for an (N, C) time series:
    for each point, use the local mean and standard deviation over window/2 steps
    on either side to form the interval [mean - n_sigma*std, mean + n_sigma*std]
    and clip extreme values.
    NaNs are ignored in local statistics (replaced by the column mean so that
    extreme points do not dominate); NaNs remain NaN after clipping.
    """
    T = np.array(T, dtype=np.float32).copy()
    N, C = T.shape
    half = window // 2
    kernel = np.ones(window, dtype=np.float32)
    for c in range(C):
        col = T[:, c]
        nan_mask = np.isnan(col)
        col_f = np.where(nan_mask, np.nanmean(col) if (~nan_mask).any() else 0.0, col)
        sums = np.convolve(col_f, kernel, mode='same')
        counts = np.convolve(np.ones_like(col_f), kernel, mode='same')
        means = sums / np.maximum(counts, 1.0)
        sums2 = np.convolve(col_f ** 2, kernel, mode='same')
        stds = np.sqrt(np.clip(sums2 / np.maximum(counts, 1.0) - means ** 2, 0, None))
        lo = means - n_sigma * stds
        hi = means + n_sigma * stds
        clipped = np.clip(col, lo, hi)
        clipped = np.where(nan_mask, np.nan, clipped)
        T[:, c] = clipped
    return T

def fit_standardizer(T):
    """
    Fit z-score parameters from T (N, C). Return (mean, std), both of shape (1, C).
    Call only on the training set to avoid using validation/test information.
    """
    mean = np.nanmean(T, axis=0, keepdims=True)
    std  = np.nanstd(T,  axis=0, keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform(T, mean, std):
    """Standardize any segment using the mean/std fitted on the training set."""
    return ((T - mean) / std).astype(np.float32)


def make_windows(T, window_size, horizon, stride=1, target_idx=None, drop_nan=False):
    """
    Slide a window over the time series to construct (X, y).

    Parameters
    ----------
    T           : (N, C), already standardized
    window_size : input length
    horizon     : forecast horizon (number of steps)
    stride      : sliding window stride
    target_idx  : None means predict all C variables; an int means predict only that column

    Returns
    -------
    X : (M, window_size, C)
    y : (M, horizon, C)        when target_idx=None
        or (M, horizon)        when target_idx=int
    """
    N = T.shape[0]
    last_start = N - window_size - horizon
    if last_start < 0:
        raise ValueError(
            f'Sequence length ({N}) is insufficient to construct samples'
        )
    starts = np.arange(0, last_start + 1, stride)

    X = np.stack([T[s:s + window_size] for s in starts], axis=0)
    Y = np.stack([T[s + window_size:s + window_size + horizon] for s in starts], axis=0)

    if target_idx is not None:
        Y = Y[..., target_idx]     # (M, horizon)

    if drop_nan:
        x_nan = np.isnan(X).any(axis=(1, 2))
        y_nan = np.isnan(Y).any(axis=tuple(range(1, Y.ndim)))
        keep = ~(x_nan | y_nan)
        X, Y = X[keep], Y[keep]

    return X.astype(np.float32), Y.astype(np.float32)


def make_loader(X, y, batch_size=64, shuffle=True):
    """Wrap numpy (X, y) into a DataLoader."""
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def split_time_series(T, train_ratio=0.7, val_ratio=0.15, gap=0):
    """
    Split in chronological order and return (T_train, T_val, T_test).
    A time series must not be split randomly, otherwise future information leaks.

    gap: leave an isolation band between training and validation (e.g. WINDOW_SIZE+HORIZON);
    samples inside the band belong to no subset, avoiding overfitting-style over-prediction
    caused by "start of validation = tail of training". Raise an error if there is not
    enough data to carve out a validation set after the gap; if the full val_ratio cannot
    fit, shrink the validation set so that a test segment still remains.
    """
    N = len(T)
    n_train = int(N * train_ratio)
    start_val = n_train + max(0, gap)
    if start_val >= N:
        raise ValueError(
            f"Sequence length ({N}) is insufficient to split a validation set. after gap={gap}. Please reduce the gap or increase the data size."
        )
    n_val = int(N * val_ratio)
    end_val = start_val + n_val
    if end_val > N:
        n_val = max(0, N - start_val)
        end_val = start_val + n_val
    T_train = T[:n_train]
    T_val   = T[start_val:end_val]
    T_test  = T[end_val:]
    return T_train, T_val, T_test


def fit_clip_bounds(T_train, q_low=1.0, q_high=99.0):
    """Fit clipping bounds from training-set global quantiles; return (lo, hi), both of shape (C,).
    All-NaN columns fall back to +/-inf (i.e. no clipping)."""
    lo = np.nanpercentile(T_train, q_low,  axis=0).astype(np.float32)
    hi = np.nanpercentile(T_train, q_high, axis=0).astype(np.float32)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi),  np.inf, hi)
    return lo, hi


def apply_clip(T, lo, hi):
    """Clip an (N, C) array column-wise to [lo, hi]; NaNs remain NaN."""
    return np.clip(T, lo, hi).astype(np.float32)
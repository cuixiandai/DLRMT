from copy import deepcopy


# ---------- Timestep conversion ----------
# The history window is uniformly converted from 7 days (steps depend on sampling
# interval: 30min -> 336, 1h -> 168);
# the forecast horizon is set per dataset (e.g. BaffleCreek/Burnett/Guadalupe use
# 0.5 day = 24 steps), not uniformly 7 days.
_7D_30MIN = 7 * 24 * 2      # 30 min sampling -> 336 steps
_7D_1H    = 7 * 24          # 1 h  sampling -> 168 steps
_3D_30MIN = 3 * 24 * 2
_1D_30MIN = 1 * 24 * 2
_0p5D_30MIN =  24 


# ---------- Training hyperparameters (shared by all datasets, configured here) ----------
# Learning rate, weight decay, early stopping and other training hyperparameters are
# managed centrally here for reproducibility.
_GLOBAL = {
    'learning_rate': 1e-4,     # AdamW learning rate
    'weight_decay':  1e-4,     # weight decay (regularization to suppress overfitting)
    'batch_size':    128,
    'max_epoch':     200,      # maximum number of training epochs
    'min_epoch':     100,      # minimum epoch guard before early stopping
    'patience':      50,       # early-stopping patience (stop after N epochs without validation improvement)
    'stride':        1,
    'train_ratio':   0.7,
    'val_ratio':     0.1,
    'target_idx':    None,
    'output_model':  '/tmp/output/model.pth',
}


# ---------- Dataset-specific parameters ----------
# The paper experiments include only BaffleCreek / Burnett / Guadalupe;
# the rest are reserved datasets and do not participate in the paper's result comparisons.
_DATASETS = {
    'BaffleCreek': {
        'n_features':  5,
        'var_names':   ['Temp', 'pH', 'Turbidity', 'Chl a', 'DO'],
        'sampling':    '30min',
        'window_size': _7D_30MIN,
        'horizon':     _0p5D_30MIN,
        # Actual range of the current CSV, not the paper's range
        'data_start': '2014-01-01 00:00:30',
        'data_end': '2014-12-18 23:30:30',
        'paper_start': '2013-11-01',
        'paper_end':   '2014-07-29',
        'paper_records': 12992,
    },

    'Burnett': {
        'n_features':  5,
        'var_names':   ['Temp', 'pH', 'Turbidity', 'Chl a', 'DO'],
        'sampling':    '30min',
        'window_size': _7D_30MIN,
        'horizon':     _0p5D_30MIN,
    },
    'Guadalupe': {
        'n_features':  4,
        'var_names':   ['Sal', 'DO', 'pH', 'Temp'],
        'sampling':    '30min',
        'window_size': _7D_30MIN,
        'horizon':     _0p5D_30MIN,
    },
    'FHLOO': {
        'n_features':  5,
        'var_names':   ['DO', 'CO2', 'pH', 'Temp', 'Sal'],
        'sampling':    '1h',
        'window_size': _7D_1H,
        'horizon':     _7D_1H,
    },
    'Gregory': {
        'n_features':  5,
        'var_names':   ['Chl a', 'DO', 'pH', 'Temp', 'Turb'],
        'sampling':    '30min',
        'window_size': _7D_30MIN,
        'horizon':     _7D_30MIN,
    },
    'Shandong1': {
        'n_features':  3,
        'var_names':   ['Temp', 'DO', 'Chl a'],
        'sampling':    '1h',
        'window_size': _7D_1H,
        'horizon':     _7D_1H,
    },
    'Shandong2': {
        'n_features':  2,
        'var_names':   ['Temp', 'DO'],
        'sampling':    '1h',
        'window_size': _7D_1H,
        'horizon':     _7D_1H,
    },
}


def get_config(name):
    """Return the full config dict for `name` (global + dataset-specific)."""
    if name not in _DATASETS:
        raise KeyError(
            f"Dataset '{name}' is not registered in config.py. "
            f"Registered datasets: {list(_DATASETS.keys())}"
        )
    cfg = deepcopy(_GLOBAL)
    cfg.update(deepcopy(_DATASETS[name]))
    cfg['name'] = name
    return cfg


def list_datasets():
    return list(_DATASETS.keys())
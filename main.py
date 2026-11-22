import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from load_data import load_data
from utils import *
from model import MyModel
from config import get_config
import random

DATA_NAME     = 'BaffleCreek'
COL_INDEX     = -1        # -1: all columns; >=0: use only the single column at this index
EVAL_EPOCH    = 0         # start evaluation after epoch 0
random_state=345
random.seed(random_state)
cfg = get_config(DATA_NAME)

# Read the learning rate from config, after cfg has been defined
LEARNING_RATE = cfg['learning_rate']

WINDOW_SIZE  = cfg['window_size']
HORIZON      = cfg['horizon']
STRIDE       = cfg['stride']
BATCH_SIZE   = cfg['batch_size']
TRAIN_RATIO  = cfg['train_ratio']
VAL_RATIO    = cfg['val_ratio']
TARGET_IDX   = cfg['target_idx']
NUM_FEATURES = cfg['n_features']
MAX_EPOCH    = cfg['max_epoch']
MIN_EPOCH    = cfg['min_epoch']
PATIENCE     = cfg['patience']
WEIGHT_DECAY = cfg['weight_decay']

# Assert early in the main flow to prevent misconfigured values
assert MIN_EPOCH <= MAX_EPOCH, \
    f'min_epoch={MIN_EPOCH} cannot be larger than max_epoch={MAX_EPOCH}'
OUTPUT_MODEL = cfg['output_model']
VAR_NAMES    = cfg['var_names']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#criterion = nn.MSELoss()
criterion = nn.HuberLoss(delta=1.0)   # Huber loss for training (robust to high-frequency noise/outliers); evaluation still follows the paper with MAE/RMSE/R2/KGE

best_score = float('inf')

print_seg=False
print_phy=False
print_len=False
# ----------------------------- Functions -----------------------------
def save(model, optimizer):
    out_dir = os.path.dirname(OUTPUT_MODEL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, OUTPUT_MODEL)


def eval(model, criterion, data_loader, n_segments=5):
    """Compute the average validation MSE and print diagnostics:
      - overall mean/std comparison of predictions vs. ground truth
        (pred_mean noticeably greater than gt_mean means "overheating");
      - split the validation windows chronologically into n_segments segments and
        report MSE and mean comparison for each."""
    print('\033[1;35m----Evaluating----\033[0m')
    model.eval()
    total_loss, nb_batches = 0.0, 0
    preds, gts = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item()
            nb_batches += 1
            preds.append(pred.cpu().numpy())
            gts.append(batch_y.cpu().numpy())

    avg_loss = total_loss / max(nb_batches, 1)
    preds = np.concatenate(preds, axis=0)
    gts   = np.concatenate(gts,   axis=0)

    p = preds.reshape(-1)
    g = gts.reshape(-1)
    print(f'Validation MSE: {avg_loss:.6f}')
    print(f'  pred mean/std = {p.mean():.4f}/{p.std():.4f}  '
          f'gt mean/std = {g.mean():.4f}/{g.std():.4f}')
    # Additional quantile diagnostics to expose train/val/test distribution shifts
    for c, name in enumerate(VAR_NAMES):
        pp = preds[..., c].reshape(-1)
        gg = gts[..., c].reshape(-1)
        print(f'  {name:>10s}: pred p1/p50/p99 = '
              f'{np.percentile(pp,1):7.2f}/{np.percentile(pp,50):7.2f}/{np.percentile(pp,99):7.2f}  | '
              f'gt p1/p50/p99 = '
              f'{np.percentile(gg,1):7.2f}/{np.percentile(gg,50):7.2f}/{np.percentile(gg,99):7.2f}')
    if p.mean() > g.mean() + 1e-3:
        print('  \033[1;33m⚠ The predictions are systematically biased high (possibly still "overheating")\033[0m')

    # Split chronologically (windows are built with increasing start indices, so order equals time order)
    M = preds.shape[0]
    if M >= n_segments:
        for i, idx in enumerate(np.array_split(np.arange(M), n_segments)):
            pp = preds[idx].reshape(-1)
            gg = gts[idx].reshape(-1)
            seg_mse = float(np.mean((pp - gg) ** 2))
            if print_seg:
                print(f'  seg{i}: n={len(idx):4d}  MSE={seg_mse:.4f}  '
                      f'pred_mean={pp.mean():.4f}  gt_mean={gg.mean():.4f}')
    model.train()
    return avg_loss


def train(model, optimizer, criterion, train_loader, val_loader):
    """
    Early stopping triggers when both conditions hold:
      1) current epoch count >= MIN_EPOCH (minimum-epoch guard)
      2) validation MSE has not decreased for PATIENCE consecutive epochs
    """
    global best_score
    print('-----------------training------------')
    no_improve = 0

    for epoch in range(MAX_EPOCH):
        print('【Epoch {}/{}】best loss = {:.6f}📊'.format(
            epoch + 1, MAX_EPOCH, best_score))

        # --- Train for one epoch ---
        for batch_x, batch_y in tqdm(train_loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch < EVAL_EPOCH:
            continue

        # --- Validation ---
        val_loss = eval(model, criterion, val_loader)

        # The (epoch + 1) >= MIN_EPOCH restriction has been removed:
        # save the model as soon as validation loss hits a new low (even at epoch 1).
        if val_loss < best_score:
            best_score = val_loss
            no_improve = 0
            save(model, optimizer)
            print('\033[1;31m Better model saved.\033[0m')
        else:
            no_improve += 1

        # --- Early stopping: keep the MIN_EPOCH guard to prevent an accidental
        # early low value from stopping training too soon ---
        if (epoch + 1) >= MIN_EPOCH and no_improve >= PATIENCE:
            print(f'Early stopping at epoch {epoch + 1} '
                  f'(min_epoch={MIN_EPOCH}, patience={PATIENCE}, '
                  f'best={best_score:.6f})')
            break


def _r2(p, g):
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _kge(p, g):
    r = np.corrcoef(p, g)[0, 1]
    mu_p, mu_g = p.mean(), g.mean()
    sd_p, sd_g = p.std(),  g.std()
    if mu_g == 0 or sd_g == 0:
        return float('nan')
    return 1.0 - np.sqrt((r - 1) ** 2 +
                         (sd_p / sd_g - 1) ** 2 +
                         (mu_p / mu_g - 1) ** 2)


def test(model, data_loader, mean, std):
    """
    Evaluate on the test set: collect predictions and ground truth ->
    first print metrics in standardized units (tab-separated table),
    then unnormalize to physical units for output (if print_phy is True).
    Table format:
        Metrics   <var1>  <var2> ...  Overall
        MAE       ...
        RMSE      ...
        R2        ...
        KGE       ...
    """
    print('\033[1;35m----Testing----\033[0m')
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            preds.append(pred)
            gts.append(batch_y.numpy())

    preds = np.concatenate(preds, axis=0)   # (M, horizon, C)
    gts   = np.concatenate(gts,   axis=0)

    def _compute_and_print(preds, gts, title):
        """Compute per-variable and overall metrics, then print them as a tab-separated table."""
        print(f'\n=== {title} ===')
        per_var = []
        for c in range(preds.shape[-1]):
            p = preds[..., c].reshape(-1)
            g = gts[..., c].reshape(-1)
            per_var.append(dict(
                MAE =float(np.mean(np.abs(p - g))),
                RMSE=float(np.sqrt(np.mean((p - g) ** 2))),
                R2  =float(_r2(p, g)),
                KGE =float(_kge(p, g)),
            ))

        p_all = preds.reshape(-1)
        g_all = gts.reshape(-1)
        overall = dict(
            MAE =float(np.mean(np.abs(p_all - g_all))),
            RMSE=float(np.sqrt(np.mean((p_all - g_all) ** 2))),
            R2  =float(_r2(p_all, g_all)),
            KGE =float(_kge(p_all, g_all)),
        )

        # Header: Metrics  var1  var2 ...  Overall
        headers = list(VAR_NAMES) + ['Overall']
        print('Metrics\t' + '\t'.join(headers))
        # One row per metric
        for m in ('MAE', 'RMSE', 'R2', 'KGE'):
            row = [m] + [f'{d[m]:.4f}' for d in per_var] + [f'{overall[m]:.4f}']
            print('\t'.join(row))
        return overall

    # 1) Table in standardized units (following the common convention of Table 2 in the paper)
    _compute_and_print(preds, gts, 'Test results (standardized units)')

    # Unnormalize (mean/std have shape (1, C) and broadcast automatically)
    preds_phys = preds * std + mean
    gts_phys   = gts   * std + mean

    # 2) Table in physical units (for practical interpretation)
    '''
    if print_phy:
        overall = _compute_and_print(preds_phys, gts_phys,
                                     'Test results (physical units)')
    '''

    # Even if physical-unit tables are not printed, return values in physical units
    # (more meaningful for reporting)
    p_all = preds_phys.reshape(-1)
    g_all = gts_phys.reshape(-1)
    overall = dict(
        MAE =float(np.mean(np.abs(p_all - g_all))),
        RMSE=float(np.sqrt(np.mean((p_all - g_all) ** 2))),
        R2  =float(_r2(p_all, g_all)),
        KGE =float(_kge(p_all, g_all)),
    )

    model.train()
    return overall['MAE'], overall['RMSE'], overall['R2'], overall['KGE']


# ----------------------------- Main flow -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--col', type=int, default=COL_INDEX,
                        help='Column index: -1 for all columns; >=0 for a single column at the specified index')
    args = parser.parse_args()
    col_index = args.col

    print('start')
    print(f'dataset: {DATA_NAME}, col={col_index}')
    print(f'window={WINDOW_SIZE}, horizon={HORIZON}, batch={BATCH_SIZE}, '
          f'epoch={MAX_EPOCH}, min_epoch={MIN_EPOCH}, '
          f'lr={LEARNING_RATE}, patience={PATIENCE}')

    # 1) Load data (N, C), filtering the range by data_start/data_end from config
    T = load_data(DATA_NAME, cfg.get('data_start'), cfg.get('data_end'))

    # ---- Single-column mode: slice data + sync global config ----
    if col_index >= 0:
        if col_index >= T.shape[1]:
            raise IndexError(
                f'--col {col_index} exceeds the number of data columns ({T.shape[1]})'
            )
        T = T[:, col_index:col_index + 1]          # keep 2D: (N, 1)

        NUM_FEATURES = 1
        VAR_NAMES    = [cfg['var_names'][col_index]]
        cfg['n_features'] = 1
        cfg['var_names']  = VAR_NAMES

        stem, ext = os.path.splitext(OUTPUT_MODEL)
        OUTPUT_MODEL = f'{stem}_col{col_index}{ext}'

        print(f'[col] use single column index={col_index} ({VAR_NAMES[0]}), '
              f'n_features=1, ckpt={OUTPUT_MODEL}')
    if print_len:
        print('raw T shape       :', T.shape, T.dtype)
        # Instead of hard-coding the 12992 assertion, print the length and warn
        print(f'Actual data length: {len(T)}')
        if len(T) != cfg.get('paper_records', len(T)):
            print('Warning: The current data length does not match the paper; metrics are not directly comparable.')
        assert T.shape[1] == NUM_FEATURES, (
            f"NUM_FEATURES ({NUM_FEATURES}) does not match the number of data columns ({T.shape[1]}). Please verify 'n_features' in config.py."
        )

    # 2) First split chronologically (with a gap isolation band), then process each subset
    # independently, preventing training-tail statistics from leaking into validation/test starts.
    GAP = WINDOW_SIZE + HORIZON   # at least 672 steps of isolation
    T_train, T_val, T_test = split_time_series(
        T, TRAIN_RATIO, VAL_RATIO, gap=GAP)
    print(f'split (gap={GAP}): train={len(T_train)}, '
          f'val={len(T_val)}, test={len(T_test)}')

    # 3) Handle missing values per subset + global clipping based on training p1/p99.
    #    fill_boundary=False: keep leading/trailing NaNs to avoid cross-subset contamination
    T_train = fill_missing(T_train, fill_boundary=False)
    T_val   = fill_missing(T_val,   fill_boundary=False)
    T_test  = fill_missing(T_test,  fill_boundary=False)
    lo, hi = fit_clip_bounds(T_train, q_low=1.0, q_high=99.0)
    T_train = apply_clip(T_train, lo, hi)
    T_val   = apply_clip(T_val,   lo, hi)
    T_test  = apply_clip(T_test,  lo, hi)
    print('[preprocess] clip bounds (from train p1/p99):')
    for c, name in enumerate(VAR_NAMES):
        print(f'  {name:>10s}: [{lo[c]:8.2f}, {hi[c]:8.2f}]')
    print('after fill+clip | remaining NaN:',
          f'train={int(np.isnan(T_train).sum())} '
          f'val={int(np.isnan(T_val).sum())} '
          f'test={int(np.isnan(T_test).sum())}')

    # 4) Fit standardization parameters on the training set only
    mean, std = fit_standardizer(T_train)
    T_train = transform(T_train, mean, std)
    T_val   = transform(T_val,   mean, std)
    T_test  = transform(T_test,  mean, std)

    # 5) Sliding windows
    #    Training set: overlapping windows + drop windows containing NaNs
    #    (NaNs may remain at the leading/trailing boundaries)
    X_train, y_train = make_windows(
        T_train, WINDOW_SIZE, HORIZON, STRIDE, TARGET_IDX, drop_nan=True)
    #    Validation/test sets use non-overlapping windows (stride related to horizon),
    #    so MSE is estimated from independent samples and high overlap does not
    #    amplify the error of a single segment
    val_stride = max(1, HORIZON // 7)
    X_val,   y_val   = make_windows(
        T_val,   WINDOW_SIZE, HORIZON, val_stride, TARGET_IDX, drop_nan=True)
    X_test,  y_test  = make_windows(
        T_test,  WINDOW_SIZE, HORIZON, val_stride, TARGET_IDX, drop_nan=True)

    print('X_train:', X_train.shape, 'y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   'y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  'y_test :', y_test.shape)

    # 6) DataLoaders
    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    # 7) Model / optimizer
    model = MyModel(
        n_features=cfg['n_features'],
        window_size=cfg['window_size'],
        horizon=cfg['horizon'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,      # 1e-3
    )

    # 8) Train (save the best model on validation, with early stopping)
    train(model, optimizer, criterion, train_loader, val_loader)
    print(f'Best validation loss: {best_score:.6f}')

    # 9) Load the best model -> evaluate on the test set
    if os.path.exists(OUTPUT_MODEL):
        ckpt = torch.load(OUTPUT_MODEL, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Loaded best model from', OUTPUT_MODEL)
    else:
        print('Warning: Optimal model file not found; the current model will be used for evaluation.')

    test(model, test_loader, mean, std)
## Dynamic Low-Rank Mamba–Transformer with Cross-Channel Fusion for High-Frequency Multivariate Water-Quality Forecasting

## Abstract

Time-series forecasting of high-frequency aquaculture water quality, characterized by non-stationarity, noise dominance, and time-varying cross-variable coupling, is a critical task. Its results directly support hypoxia and harmful algal bloom early warnings as well as production-control decisions such as aeration and feeding. In recent years, deep learning methods have gradually supplanted traditional statistical and mechanistic models to become the mainstream paradigm for water-quality time-series forecasting. However, existing deep learning approaches still suffer from two inherent limitations. First, mainstream Transformers rely on self-attention, whose quadratic complexity forces long sequences to be truncated into windows, making it difficult to capture the diurnal and tidal long-range rhythms of water-quality series. Meanwhile, self-attention maintains a fully connected topology along the variable axis, allowing noise to propagate indiscriminately across channels along spurious-correlation edges. Since normalization merely redistributes edge weights without severing edges topologically, such leakage cannot be structurally blocked. Second, temporal-axis memory and variable-axis sparse coupling are fundamentally different in nature yet are modeled jointly. Existing low-rank approaches typically treat the rank as a fixed hyperparameter and place the compression bottleneck on the time axis, ignoring the fact that water-quality redundancy arises mainly across variables. This forces denoising and detail preservation to be traded off at a single operating point. To address these issues, this paper proposes the Dynamic Low-Rank Mamba-Transformer (DLRMT), which decomposes the problem along the two axes inherent to the hydrochemical system. It adopts Mamba, with its linear complexity and input-dependent selective state transitions, as the temporal backbone to adaptively retain or forget long-range temporal memory without truncating the sequence, thereby fundamentally circumventing the windowing cost of self-attention. Furthermore, it delegates the sparse, directed, and highly redundant cross-variable coupling to a structure-constrained low-rank attention path. Experiments on three datasets, namely Baffle Creek, Burnett, and Guadalupe Estuary, show that DLRMT outperforms the state-of-the-art baselines across MAE, RMSE, R2, and KGE. 

## Requirements:

- Python 3.11
- PyTorch >= 2.10.0
- mamba-ssm >= 2.3.0

## Usage:

python main.py


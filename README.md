# Spectral Analysis of Representational Similarity with Limited Neurons

Code accompanying the NeurIPS 2025 paper [*Spectral Analysis of Representational Similarity with Limited Neurons*](https://openreview.net/forum?id=sfTtFZXINg) by Hyunmo Kang, Abdulkadir Canatar, and SueYeon Chung.

The paper applies Random Matrix Theory (RMT) to study how the finite number of recorded neurons biases representational similarity measures such as Centered Kernel Alignment (CKA) and Canonical Correlation Analysis (CCA). It derives analytical predictions relating measured similarity to the spectrum and eigenvector overlaps of the population representations, shows that only on the order of the square root of the number of recorded neurons contributes to localized modes under power-law spectra, and introduces a denoising estimator that recovers population-level similarity from small samples.

## Repository layout

Top-level Jupyter notebooks reproduce the figures in the paper:

- `fig1.ipynb` — illustrative example of finite-neuron bias in CKA/CCA.
- `fig2_fig3.ipynb`, `fig3_localized_delocalized.ipynb` — spectral theory predictions and the localized/delocalized eigenvector transition.
- `fig4_cca_flip.ipynb` — CCA behavior and sign/ordering effects under subsampling.
- `fig5_brainscore_exp.ipynb` — experiments on neural datasets (Brain-Score, etc.).
- `figSI1.ipynb` — supplementary figure.
- `dnn_activations.ipynb` — extracting deep network activations used by the other notebooks.

The `src/` package contains the supporting library (see below). Generated activations, datasets, results, and figures are written to local `activations/`, `datasets/`, `results/`, and `figures/` folders (gitignored).

## The `src/` package

Core theory and estimators:

- `spectral_cka.py` — main entry point. Computes Gram-matrix spectra, population eigenvector overlap matrices, true/naive/predicted/denoised CKA and CCA, and averages them over random neuron subsamplings. Implements the paper's RMT-based prediction and denoising pipeline.
- `rmt_utils.py` — RMT primitives used by the theory: iterative solvers for the Stieltjes and Silverstein transforms, and computation of the theoretical self-overlap matrix `Q` between population and sample eigenvectors.
- `cka_utils.py` — unbiased moment estimators of the numerator/denominator of CKA (naive, Kong–Valiant, and the paper's column-independence-corrected estimator) via einsum over U-statistic terms.
- `dim_utils.py` — utilities for effective dimension / participation-ratio style quantities derived from the representation spectra.
- `power_law_utils.py` — power-law covariance sampling, analytic eigenvalue predictions under power-law spectra, and routines to fit the power-law exponent from empirical spectra.

Data and models:

- `data_utils.py` — thin wrappers that load the supported neural datasets (Stringer mouse V1, Brain-Score Majaj–Hong 2015 IT/V4 and Freeman–Ziemba 2013 V1/V2, THINGS fMRI, TVSD) and convert stimulus dataframes into image tensors; also loads CIFAR-10.
- `neural_data/` — dataset-specific loaders used by `data_utils` (`neural_data_brainscore.py`, `neural_data_stringer.py`, `neural_data_tvsd.py`, `neural_data_things_fmri.py`).
- `model_utils.py` — extracts and caches activations from `timm` vision models over a given image set, with a `torch.utils.data.Dataset` for dataframes of images.
- `llm_tools/` — optional helpers for LLM-based experiments: a registry of Hugging Face causal models (`llm_models.py`), prompt/extraction utilities (`extract_tools.py`), and FLORES multilingual prompts (`flores.py`).

## Citation

```bibtex
@inproceedings{
kang2026spectral,
title={Spectral Analysis of Representational Similarity with Limited Neurons},
author={Hyunmo Kang and Abdulkadir Canatar and SueYeon Chung},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=sfTtFZXINg}
}
```

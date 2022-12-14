# KrADagrad

Official PyTorch implementation of Kronecker Approximation-Domination preconditioned gradient optimization

- Main code lives in `kradagrad.py` (KrADagrad* in the paper) and `kradagradmm.py` (KrADagrad in the paper)
- We include `shampoo` in `third_party/` with minor modifications for convenience
- An implementation of resnet also lives in `third_party/` for convenience
- Methods for taking p-th roots (and other useful utilities) for symmetric positive semi-definite matrices live in `positive_matrix_functions.py`
- Experiments (training notebooks+scripts) will be found in `experiments/`
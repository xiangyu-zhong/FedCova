# FedCova

Minimal release for one CIFAR-10 experiment: non-i.i.d., 20 clients, full participation, 500 rounds, symmetric noise `(rho, tau) = (0.6, 0.7)`, data/noise seed 2, training seed 11, `e = 2`, and classifier `alpha = 2`.

## Run

Use Python 3.8 and a CUDA GPU. Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The dependency versions follow the original environment export (PyTorch 1.10.1 and torchvision 0.11.2, originally used with CUDA 10.2). Use mutually compatible PyTorch/torchvision CUDA builds for your system. This minimal installation has not yet been validated by a complete training run. The code does not require scikit-learn, cvxpy, or MOSEK.

Download CIFAR-10 and place the extracted files under `datasets/cifar10/cifar-10-batches-py/`. Dataset paths can be adjusted in `datasets/dataset_base.py`.

```bash
python MonteCarlo.py
```

Results are created in `output/`. The final printed mean and standard deviation describe the last five rounds of this single run. Exact numerical results may vary with software and hardware.

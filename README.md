# FedCova: Robust Federated Covariance Learning Against Noisy Labels

This repository contains the official code for the paper "FedCova: Robust Federated Covariance Learning Against Noisy Labels". This release provides a CIFAR-10 experiment with heterogeneous data and symmetric label noise.

## Authors

Xiangyu Zhong (The Chinese University of Hong Kong), Xiaojun Yuan (University of Electronic Science and Technology of China), and Ying-Jun Angela Zhang (The Chinese University of Hong Kong)

## TL;DR

FedCova unifies federated feature learning, classifier construction, and noisy-label correction through class feature covariances to improve robustness against noisy labels without relying on clean public data or clean-device selection.

## Abstract

Noisy labels in distributed datasets induce severe local overfitting and consequently compromise the global model in federated learning (FL). Most existing solutions rely on selecting clean devices or aligning with public clean datasets, rather than endowing the model itself with robustness. In this paper, we propose FedCova, a dependency-free federated covariance learning framework that eliminates such external reliances by enhancing the model’s intrinsic robustness via a new perspective on feature covariances. Specifically, FedCova encodes data into a discriminative but resilient feature space to tolerate label noise. Built on mutual information maximization, we design a novel objective for federated lossy feature encoding that relies solely on class feature covariances with an error tolerance term. Leveraging feature subspaces characterized by covariances, we construct a subspace-augmented federated classifier. FedCova unifies three key processes through the covariance: (1) training the network for feature encoding, (2) constructing a classifier directly from the learned features, and (3) correcting noisy labels based on feature subspaces. We implement FedCova across both symmetric and asymmetric noisy settings under heterogeneous data distribution. Experimental results on CIFAR-10/100 and real-world noisy dataset Clothing1M demonstrate the superior robustness of FedCova compared with the state-of-the-art methods.

## Key Files

This repository is organized as follows:

- `MonteCarlo.py`: Main entry point for data loading, federated training, evaluation, and result output.
- `browser/options_base.py`: Experiment settings, including data partitioning, noise levels, training seeds, and correction settings.
- `datasets/dataset_base.py`: CIFAR-10 loading, preprocessing, and heterogeneous client data partitioning.
- `datasets/cifar.py` and `datasets/utils.py`: Dataset implementation and label-noise utilities.
- `modules/modules_environments.py` and `modules/modules_subsystems.py`: Federated training workflow, model and covariance aggregation, and label correction.
- `modules/modules_networks.py`: Network structures and local training.
- `modules/loss.py`: Covariance-based feature learning objective.
- `modules/classifier.py`: Covariance-based classifier, with `e = 2` and `alpha = 2` for the provided experiment.
- `browser/visual_base.py`: CSV result export utilities.
- `requirements.txt`: Required Python packages.

## Installation and Usage

**Install Required Python Packages**

Use Python 3.8 and a CUDA-enabled GPU. The required packages include PyTorch, torchvision, NumPy, SciPy, pandas, matplotlib, Pillow, opt-einsum, and six.

```bash
git clone https://github.com/xiangyu-zhong/FedCova.git
cd FedCova
python -m pip install -r requirements.txt
```

The dependency versions follow the original environment export: PyTorch 1.10.1 and torchvision 0.11.2, originally used with CUDA 10.2. Use mutually compatible PyTorch/torchvision CUDA builds for your system. The minimal dependency installation has not yet been validated by a complete training run.

**Prepare CIFAR-10**

Place the extracted CIFAR-10 files under:

```text
datasets/cifar10/cifar-10-batches-py/
```

The loader also attempts to download CIFAR-10 if the files are missing. Dataset paths can be adjusted in `datasets/dataset_base.py`.

**Run the Experiment**

```bash
python MonteCarlo.py
```

The default configuration uses ResNet-18, 20 clients with full participation, 500 communication rounds, one local epoch, and symmetric noise `(rho, tau) = (0.6, 0.7)`. The data/noise seed is 2 and the training seed is 11. Both the loss parameter `e` and the classifier parameter `alpha` are set to 2.

Results are saved as CSV files in `output/`, including test accuracy, runtime, feature orthogonality, and label-noise statistics. The final printed mean and standard deviation summarize test accuracy over the last five communication rounds of the single run. Exact numerical results may vary with software and hardware.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{zhong_fedcova,
  title={FedCova: Robust Federated Covariance Learning Against Noisy Labels},
  author={Zhong, Xiangyu and Yuan, Xiaojun and Zhang, Ying-Jun Angela},
  note={Manuscript}
}
```

For questions or discussions, please contact xyzhong@ie.cuhk.edu.hk.

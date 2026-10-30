import numpy as np
import torch
from browser.options_base import DEVICETYPE

e = 2
alpha = 2
T = 10


def evaluate(state, method, train_features, train_labels, means=None, covariances=None, weights=None, subspace=None, test_features=None, test_labels=None, **kwargs):
    if method != 'gmm_map':
        raise ValueError('This release supports only the gmm_map classifier.')
    if state == 'train':
        return gmm_map(state, train_features, train_labels, **kwargs)
    if state in ('test', 'detect'):
        return gmm_map(state, train_features, train_labels, means, covariances, weights, subspace, **kwargs)
    raise ValueError('Unknown classifier state: {}'.format(state))


def gmm_map(state, train_features, train_labels, means=None, covariances=None, weights=None, subspaces=None, test_features=None, test_labels=None, num_classes=[], return_pred=False, label_presence=None, epsilon=None):
    train_features = train_features.to(DEVICETYPE)
    train_labels = train_labels.to(DEVICETYPE)
    test_features = train_features
    test_labels = train_labels
    if state == 'train':
        means = []
        covariances = []
        weights = []
        subspace = []
        for i in range(len(num_classes)):
            if num_classes[i] != 0:
                class_data = train_features[train_labels == i]
                N = class_data.size(0)
                mean = torch.zeros(class_data.size(1), device=DEVICETYPE)
                covariance = (N - 1) / N * torch_cov(class_data, mean).to(DEVICETYPE)
                means.append(mean)
                covariances.append(covariance)
                weights.append(torch.tensor(num_classes[i], device=DEVICETYPE))
            else:
                means.append(None)
                covariances.append(None)
                weights.append(None)
        train_acc = 0
        return (train_acc, means, covariances, weights, subspace)
    elif state == 'test':
        eps = e
        d = train_features.shape[1]
        num_components = len(weights)
        n = alpha
        test_scores = torch.zeros((train_features.shape[0], num_components), device=DEVICETYPE)
        log_prob_y = torch.full((train_features.shape[0], 1), -float('inf')).to(DEVICETYPE)
        for k in range(num_components):
            mean = means[k]
            covariance = covariances[k]
            weight = weights[k]
            subspace = subspaces[k]
            C_k = torch.linalg.inv(eps / d * torch.eye(d, device=DEVICETYPE) + covariance)
            test_scores[:, k] = -(train_features.double() @ torch.matrix_power(C_k, n) * train_features.double()).sum(dim=1).pow(1.0 / n)
        test_scores_prob = torch.nn.functional.softmax(test_scores / T, dim=1)
        test_pred = test_scores_prob.argmax(dim=1).to(DEVICETYPE)
        test_acc = compute_accuracy(test_pred, test_labels)
        return (test_acc, test_pred)
    elif state == 'detect':
        eps = e
        n = alpha
        d = train_features.shape[1]
        num_components = len(weights)
        test_scores = torch.zeros((train_features.shape[0], num_components), device=DEVICETYPE)
        for k in range(num_components):
            if weights[k] is not None:
                covariance = covariances[k]
                C_k = torch.linalg.inv(eps / d * torch.eye(d, device=DEVICETYPE) + covariance)
                test_scores[:, k] = -(train_features.double() @ torch.matrix_power(C_k, n) * train_features.double()).sum(dim=1).pow(1.0 / n)
        test_scores_prob = torch.nn.functional.softmax(test_scores / T, dim=1)
        test_pred = test_scores_prob.argmax(dim=1).to(DEVICETYPE)
        train_predmax = test_scores_prob[np.arange(test_scores.shape[0]), test_pred]
        train_predoriginal = test_scores_prob[np.arange(test_scores.shape[0]), train_labels.tolist()]
        test_acc = compute_accuracy(test_pred, test_labels)
        return (test_acc, test_scores_prob, train_predmax, test_pred, train_predoriginal)

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    if type(y_pred) == torch.Tensor:
        n_wrong = torch.count_nonzero(y_pred - y_true).item()
    elif type(y_pred) == np.ndarray:
        n_wrong = np.count_nonzero(y_pred - y_true)
    else:
        raise TypeError('Not Tensor nor Array type.')
    n_samples = len(y_pred)
    return (1 - n_wrong / n_samples) * 100

def torch_cov(x, x_mean):
    xm = x - x_mean
    cov_matrix = xm.T @ xm / (x.shape[0] - 1)
    return cov_matrix

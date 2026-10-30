import torch
from opt_einsum import contract

e = 2

def loss_MCR2_Cov(ZZ, h1, t):
    Z = h1
    y = t
    eps = e
    gam1 = 1
    gam2 = 1
    gamma = 0.01
    (m, d) = Z.shape
    I = torch.eye(d).to(Z.device)
    c = d / (m * eps)
    loss_expd = gam2 * logdet(gam1 * 1 / m * covariance(Z) + eps / d * I + gamma * sum(ZZ)) / 2.0
    if y.numel() == m:
        loss_comp = torch.tensor(0.0, device=Z.device)
        for j in y.unique():
            Z_j = Z[y == int(j)]
            m_j = Z_j.shape[0]
            c_j = d / (m_j * eps)
            logdet_j = logdet(1 / m_j * Z_j.T @ Z_j + eps / d * I + gamma * ZZ[j])
            loss_comp += logdet_j * m_j / (2 * m)
    return -loss_expd + loss_comp

def covariance(X):
    return contract('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    try:
        L = torch.linalg.cholesky(X)
        logdet_val = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        return logdet_val
    except:
        (sgn, logdet_val) = torch.linalg.slogdet(X)
        return sgn * logdet_val

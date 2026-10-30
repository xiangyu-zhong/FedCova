import numpy as np
import torch

def tensor2npcplx(a):
    N_temp = int(a.shape[0] / 2)
    N = int(np.ceil(a.shape[0] / 2))
    index_even = torch.arange(0, a.shape[0], 2, dtype=torch.long)
    index_odd = torch.arange(1, a.shape[0], 2, dtype=torch.long)
    b = np.zeros(N, dtype=complex)
    b.real[:N] = a[index_even]
    b.imag[:N_temp] = a[index_odd]
    return b

def all_choice(args, optimizer):
    for q in range(args.Q):
        optimizer.x_slct[q] = np.ones(args.users_num[q], dtype=float)
    return optimizer

def softcorrect(train_scores, correct_list):
    N = len(correct_list)
    probs = train_scores[correct_list]
    random_values = np.random.rand(N, 1)
    cumulative_probs = np.cumsum(probs, axis=1)
    corrected_new = (cumulative_probs > random_values).argmax(axis=1)
    return corrected_new

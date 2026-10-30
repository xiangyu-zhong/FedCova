import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
from .cifar import CIFAR10

def DatasetGet(args):
    dataset = {}
    for (task_num, task) in enumerate(args.task_list):
        data_path = './datasets/cifar10'
        args.num_classes = 10
        args.classnum = args.num_classes
        dataset['cifar10_train_origin'] = CIFAR10(root='./datasets/cifar10/', download=True, train=True, transform=transforms.Compose([transforms.RandomCrop(32, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor()]), noise_type=args.noise_type, noise_rate=args.noise_rate, softPi=args.softPi, globalnoise=args.globalnoise)
        dataset['cifar10_test_origin'] = CIFAR10(root='./datasets/cifar10/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]), noise_type=args.noise_type, noise_rate=args.noise_rate)
        Km = np.round(args.datasets_num[task_num] / args.users_num[task_num]) * np.ones(args.users_num[task_num])
        Km = Km.astype(np.int64)
        (dataset[task + '_train_users'], Km) = non_iid_dirichlet_sampling(args, task, Km, args.K[task_num], dataset, args.classnum, args.Bernoulli, args.alpha_dirichlet)
        args.datasets_num_dvc.append(Km)
    return (dataset, args)

def non_iid_dirichlet_sampling(args, task, Km, K, dataset, num_classes, p, alpha_dirichlet=100):
    rng = np.random.default_rng(args.Dataseed)
    Q = num_classes
    M = len(Km)
    Km_sum = np.sum(Km)
    dataset_temp = dataset[task + '_train_origin']
    labels = np.array(dataset_temp.train_labels)
    indices = np.arange(len(dataset_temp))
    class_indices = {q: indices[np.where(labels == q)[0]] for q in range(Q)}
    dataset_user = []
    Phi = rng.binomial(1, p, size=(M, Q))
    num_clients_per_class = np.sum(Phi, axis=0)
    Psi = [list(np.where(Phi[:, q] == 1)[0]) for q in range(Q)]
    dict_users = {}
    for q in range(Q):
        all_idxs = class_indices[q]
        p_dirichlet = rng.dirichlet([alpha_dirichlet] * num_clients_per_class[q])
        assignment = rng.choice(Psi[q], size=len(all_idxs), p=p_dirichlet.tolist())
        for client_k in Psi[q]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[assignment == client_k]))
            else:
                dict_users[client_k] = set(all_idxs[assignment == client_k])
    for i in range(M):
        if i in dict_users:
            merged_indices = np.array(list(dict_users[i]))
            dataset_user_temp = Subset(dataset[task + '_train_origin'], merged_indices)
            dataset_user.append(dataset_user_temp)
            Km[i] = len(merged_indices)
        else:
            dataset_user_temp = Subset(dataset[task + '_train_origin'], [])
            dataset_user.append(dataset_user_temp)
            Km[i] = 0
            print(f'Warning: User {i} has no data assigned!')
    return (dataset_user, Km)

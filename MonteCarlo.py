from pathlib import Path
import os
import random
import time
from browser.options_base import args_parser

def main():
    args = args_parser()
    os.chdir(Path(__file__).resolve().parent)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import pandas as pd
    import torch
    import modules.modules_environments as mm_env
    import datasets.dataset_base as data_base
    import browser.visual_base as vs_base
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires a CUDA-enabled PyTorch installation and GPU.')
    if args.method != 0 or args.userselection != 0:
        raise ValueError('This release supports noiseless, full-participation training.')
    seed = args.seedNotMC
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = Path('output')
    for name in ('time', 'accuracy', 'orthogonality', 'noisy'):
        (output / name).mkdir(parents=True, exist_ok=True)
    environment = mm_env.Environments(args)
    (dataset, args) = data_base.DatasetGet(args)
    environment.Environments_Init(dataset)
    print(args, flush=True)
    (elapsed, accuracy, orthogonality, noisy) = ([], [], [], [])
    for epoch in range(args.epochs):
        print(f'Seed: {seed}, round: {epoch + 1}/{args.epochs}', flush=True)
        start = time.time()
        environment.Environments_Iter(epoch)
        elapsed.append(time.time() - start)
        environment.Environments_Test(epoch)
        pd.DataFrame(elapsed).to_csv(output / 'time' / f'time{seed}.csv', index=True)
        accuracy = vs_base.pd_one_epoch_to_csv(args, environment, accuracy, 'acc', epoch, 1, np.ones(args.Q), PATH=str(output / 'accuracy' / f'accuracy{seed}.csv'))
        orthogonality = vs_base.pd_one_epoch_to_csv(args, environment, orthogonality, 'ortho', epoch, 1, np.ones(args.Q), PATH=str(output / 'orthogonality' / f'orthogonality{seed}.csv'))
        noisy = vs_base.pd_one_epoch_to_csv(args, environment, noisy, 'noisy', epoch, 1, np.ones(args.Q), PATH=str(output / 'noisy' / f'noisy{seed}.csv'))
    tail = np.asarray(accuracy[-5:], dtype=float).reshape(-1)
    print(f'Last-five accuracy: {tail.mean():.2f} +/- {tail.std(ddof=0):.2f}%', flush=True)
if __name__ == '__main__':
    main()

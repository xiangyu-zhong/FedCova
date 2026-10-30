import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead. Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root)))
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories

def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(filter(lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix), os.listdir(root)))
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=None, rng=None):
    """mistakes:
        flip in the pair (adjacent classes in the actual class list)
    """
    if isinstance(nb_classes, (list, np.ndarray)):
        original_classes = np.array(nb_classes)
        nb_classes_count = len(nb_classes)
    else:
        original_classes = np.arange(nb_classes)
        nb_classes_count = nb_classes
    next_class_map = {}
    for i in range(nb_classes_count):
        current_class = int(original_classes[i])
        next_class = int(original_classes[(i + 1) % nb_classes_count])
        next_class_map[current_class] = next_class
    y_train_noisy = y_train.copy()
    n = noise
    n_train = len(y_train)
    n_flip = int(n_train * noise)
    if n_flip > 0:
        if rng is not None:
            flip_indices = rng.choice(n_train, size=n_flip, replace=False)
        else:
            flipper = np.random.RandomState(random_state)
            flip_indices = flipper.choice(n_train, size=n_flip, replace=False)
        for idx in flip_indices:
            current_label = int(y_train[idx])
            y_train_noisy[idx] = next_class_map[current_label]
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
    else:
        actual_noise = 0.0
    return (y_train_noisy, actual_noise)

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=None):
    """mistakes:
        flip in the symmetric way
    """
    classes = nb_classes
    nb_classes = len(nb_classes)
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = n / (nb_classes - 1) * P
    if n > 0.0:
        P[0, 0] = 1.0 - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1.0 - n
        P[nb_classes - 1, nb_classes - 1] = 1.0 - n
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)
    return (y_train, actual_noise)

def noisify_random(y_train, noise, random_state=None, nb_classes=None, rng=None):
    y_train_noisy = y_train.copy()
    n_train = len(y_train)
    n_rand = int(n_train * noise)
    if rng is not None:
        randomize_indices = rng.choice(n_train, size=n_rand, replace=False)
        random_labels = rng.choice(nb_classes, size=n_rand, replace=True)
    else:
        randomize_indices = np.random.choice(n_train, size=n_rand, replace=False)
        random_labels = np.random.choice(nb_classes, size=(n_rand,), replace=True)
    y_train_noisy[randomize_indices] = random_labels
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise >= 0.0
    return (y_train_noisy, actual_noise)

def noisify_random2(y_train, noise, random_state=None, nb_classes=10):
    """Randomly corrupt labels in y_train.

    Parameters:
        y_train (np.ndarray): Original labels of the training data.
        noise (float): Ratio of labels to be corrupted. 0 to corrupt no labels; 1 to corrupt all labels.
        random_state (int): Random seed for reproducibility.
        nb_classes (int): Number of classes in the dataset.

    Returns:
        y_train_noisy (np.ndarray): Labels with added noise.
        actual_noise (float): Actual noise ratio applied.
    """
    if random_state is not None:
        np.random.seed(random_state)
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))
    np.fill_diagonal(P, 1 - noise)
    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    return (y_train_noisy, actual_noise)

def noisify(dataset='mnist', nb_classes=None, train_labels=None, noise_type=None, noise_rate=0, random_state=0, rng=None):
    if noise_type == 'pairflip':
        (train_noisy_labels, actual_noise_rate) = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes, rng=rng)
    if noise_type == 'symmetric':
        (train_noisy_labels, actual_noise_rate) = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'random':
        (train_noisy_labels, actual_noise_rate) = noisify_random(train_labels, noise_rate, random_state=0, nb_classes=nb_classes, rng=rng)
    return (train_noisy_labels, actual_noise_rate)

def get_concat_indices(concat_dataset):
    merged_indices = []
    current_offset = 0
    for subset in concat_dataset.datasets:
        if isinstance(subset, torch.utils.data.Subset):
            subset_indices = np.array(subset.indices)
            adjusted_indices = subset_indices + current_offset
            merged_indices.extend(adjusted_indices)
            current_offset += len(subset.dataset)
        else:
            subset_length = len(subset)
            subset_indices = np.arange(subset_length) + current_offset
            merged_indices.extend(subset_indices)
            current_offset += subset_length
    return merged_indices

import random
import numpy as np


def load_mnist_npz(mnist_file):
    """Load the MNIST dataset into memory from an NPZ.

    Parameters
    ----------
    mnist_file : str
        Path to an NPZ file of MNIST data.

    Returns
    -------
    train, valid, test: tuples of np.ndarrays
        Each consists of (data, labels), where data.shape=(N, 1, 28, 28) and
        labels.shape=(N,).
    """
    data = np.load(mnist_file)
    dsets = []
    for name in 'train', 'valid', 'test':
        x = data['x_{}'.format(name)].reshape(-1, 1, 28, 28)
        y = data['y_{}'.format(name)]
        dsets.append([x, y])

    return dsets


def shuffle_stream(obs, class_idx):
    idx = np.random.permutation(len(obs))
    count = 0
    while True:
        yield obs[idx[count]][np.newaxis, ...], class_idx
        count += 1
        if count >= len(obs):
            count = 0
            np.random.shuffle(idx)


def comparative_stream(class_streams):
    while True:
        x_in, y_true = next(random.choice(class_streams))
        x_same, _ = next(class_streams[y_true])
        y_not = y_true
        while y_not == y_true:
            y_not = random.randint(0, len(class_streams)-1)
        x_diff, _ = next(class_streams[y_not])
        yield dict(x_in=x_in, x_same=x_same, x_diff=x_diff)

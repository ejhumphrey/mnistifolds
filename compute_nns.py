"""Demonstration of building an MNIST classifier.


This script and its corresponding functionality expects the gzipped pickle file
provided courtesy of the Theano dev team at Montreal:

    http://deeplearning.net/data/mnist/mnist.pkl.gz


Sample call:
python ./examples/mnist.py \
~/mnist.pkl.gz \
./examples/sample_definition.json \
examples/sample_params.npz
"""

import argparse
# import numpy as np
import json
import sklearn.neighbors as nns
import time
import datatools


def neighbors_for_group(x_obs, orig_idxs, n_neighbors):
    """

    Returns
    -------
    nn_idxs : dict
        Original indexes in the dataset, pointing to `n_neighbors` indices.
    """
    knn = nns.NearestNeighbors(
        n_neighbors + 1, radius=1.0, algorithm='auto', leaf_size=30, p=2,
        metric='minkowski', metric_params=None, n_jobs=-1)
    x_obs = x_obs.reshape(-1, 28*28)
    knn.fit(x_obs)
    dist, pred_idxs = knn.kneighbors(x_obs)
    nn_idxs = dict()
    for oidx, nn_idx in zip(orig_idxs, pred_idxs):
        nn_idxs[oidx] = orig_idxs[nn_idx[1:]].tolist()

    return nn_idxs


def main(mnist_file, output_file, n_neighbors=20):
    train = datatools.load_mnist_npz(mnist_file)[0]
    x_obs, y_true = train
    # x_obs, y_true = x_obs[:1000], y_true[:1000]
    all_nns = dict()
    for n in range(10):
        print("[{}] Starting {}".format(time.asctime(), n))
        bidx = y_true == n
        orig_idxs = (bidx).nonzero()[0]
        nn_idxs = neighbors_for_group(x_obs[bidx], orig_idxs, n_neighbors)
        all_nns.update(**nn_idxs)

    with open(args.output_file, 'w') as fp:
        json.dump(all_nns, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("mnist_file",
                        metavar="mnist_file", type=str,
                        help="Path to the MNIST data file.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to save the classifier graph.")
    parser.add_argument("--n_neighbors",
                        metavar="--n_neighbors", type=int, default=20,
                        help="")
    args = parser.parse_args()
    main(args.mnist_file, args.output_file)

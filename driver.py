"""Demonstration of building an MNIST embedding.

Sample call
-----------
python ./driver.py \
~/mnist.npz \
model_definition.json \
param_cache.hdf5 \
--neighbor_graph.json
"""

import argparse
import biggie
import json
import optimus
import os
import pescador
import yaml

import datatools
import models


def main(dataset, param_cache, model_name, hyperparams, train_params,
         output_dir, neighbor_graph, verbose=False):

    # Create the models, and pass the trainer to a driver.
    trainer, predictor = models.pwrank(verbose)

    # Load data and configure the minibatch generator.
    train, valid, test = datatools.load_mnist_npz(os.path.expanduser(dataset))
    nn_idxs = json.load(open(os.path.expanduser(neighbor_graph)))
    cstream = datatools.neighbor_stream(train[0], nn_idxs)
    stream = pescador.buffer_batch(cstream, hyperparams.pop("batch_size"))

    # And we're off!
    output_dir = os.path.expanduser(output_dir)
    params = biggie.Stash(os.path.join(output_dir, param_cache))
    driver = optimus.Driver(name=model_name, graph=trainer,
                            parameter_cache=params)
    stats = driver.fit(stream, hyperparams=hyperparams, **train_params)
    print(stats)

    optimus.save(
        predictor, os.path.join(output_dir, "{}.json".format(model_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path to a yaml config file.")
    args = parser.parse_args()
    kwargs = yaml.load(open(args.config_file))
    main(**kwargs)

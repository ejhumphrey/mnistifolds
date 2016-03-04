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
import numpy as np
import optimus
import datatools

import models


def main(args):
    # Create the models, and pass the trainer to a driver.
    trainer, predictor = models.pwrank()
    driver = optimus.Driver(graph=trainer)

    # Load data and configure the minibatch generator.
    train, valid, test = datatools.load_mnist(args.mnist_file)
    stream = datatools.minibatch(train[0], train[1], 50)
    hyperparams = dict(learning_rate=0.02)

    # And we're off!
    stats = driver.fit(stream, hyperparams=hyperparams, max_iter=500,
                       print_freq=20)

    print(stats)
    # Note that, because they were made at the same time, `trainer` and
    #   `predictor` share the same parameters.
    y_pred = predictor(valid[0])['likelihoods'].argmax(axis=1)
    print("Accuracy: {0:0.4}".format(np.equal(valid[1], y_pred).mean()))

    # Now that it's done, save the graph and parameters to disk.
    optimus.save(predictor, args.def_file, args.param_file)

    # And, because we can, let's recreate it purely from disk.
    predictor2 = optimus.load(args.def_file, args.param_file)
    y_pred2 = predictor2(valid[0])['likelihoods'].argmax(axis=1)
    print("Consistency: {0:0.4}".format(np.equal(y_pred, y_pred2).mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("mnist_file",
                        metavar="mnist_file", type=str,
                        help="Path to the MNIST data file.")
    # Outputs
    parser.add_argument("def_file",
                        metavar="def_file", type=str,
                        help="Path to save the classifier graph.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path to save the classifier parameters.")
    main(parser.parse_args())

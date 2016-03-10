import argparse

import biggie
import time
import numpy as np
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import seaborn

import models
import datatools

FIG = plt.figure(figsize=(10, 10))
AX = FIG.gca()


def render(z_out, y_true, fps, output_file, title='', dpi=300):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title,
                    artist='Matplotlib',
                    comment='Learntest')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Create string highlighters
    palette = seaborn.color_palette("Set3", 10)
    colors = np.asarray([palette[y] for y in y_true])

    dot_params = dict(marker='o', s=50, alpha=0.75, c=colors)
    z = np.asarray(z_out)
    width = z[:, 0].max() - z[:, 0].min()
    height = z[:, 1].max() - z[:, 1].min()
    span = max([width, height])
    x_min, x_max = z[:, 0].mean() - span / 1.75, z[:, 0].mean() + span / 1.75
    y_min, y_max = z[:, 1].mean() - span / 1.75, z[:, 1].mean() + span / 1.75
    AX.set_xlim(x_min, x_max)
    AX.set_ylim(y_min, y_max)
    # print(AX.set_xlim(-max_val, max_val))
    # print(AX.set_ylim(-max_val, max_val))
    # handle.set_visible(True)

    with writer.saving(FIG, output_file, dpi):
        for frame_num, (x, y) in enumerate(z_out):
            # handle.set_offsets([x, y])
            AX.clear()
            AX.scatter(x, y, **dot_params)
            AX.set_xlim(x_min, x_max)
            AX.set_ylim(y_min, y_max)
            plt.draw()
            writer.grab_frame(pad_inches=0)
            if (frame_num % fps) == 0:
                print("[{}] Finished {} seconds."
                      "".format(time.asctime(), frame_num/float(fps)))


def main(mnist_file, param_file, fps, output_file):
    train, valid, test = datatools.load_mnist_npz(mnist_file)
    trainer, predictor = models.pwrank()
    params = biggie.Stash(param_file)

    idx = np.random.permutation(len(valid[0]))[:500]
    x_in = valid[0][idx]
    y_true = valid[1][idx]

    z_out = []
    for pkey in sorted(params.keys()):
        print("[{}] Processing - {}".format(time.asctime(), pkey))
        predictor.param_values = params.get(pkey)
        z_out += [predictor(x_in)['z_out'].T]

    render(z_out, y_true, fps, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("mnist_file",
                        metavar="mnist_file", type=str,
                        help="Filepath to the mnist data.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File path to save output movie.")
    parser.add_argument("--fps",
                        metavar="--fps", type=float, default=10.0,
                        help="Framerate for the fretmap.")
    # parser.add_argument("title",
    #                     metavar="title", type=str,
    #                     help="Title for the resulting video.")
    args = parser.parse_args()
    main(args.mnist_file, args.param_file, args.fps, args.output_file)

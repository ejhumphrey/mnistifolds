import biggie
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import numpy as np
import optimus
import seaborn

import datatools

seaborn.set()

URL_BASE = "https://raw.githubusercontent.com/ejhumphrey/mnistifolds/master/"\
           "images/valid/"


def generate_embedding(param_file, model_file, mnist_file, output_file=None,
                       param_key=None):
    predictor = optimus.load(model_file)
    params = biggie.Stash(param_file)
    param_key = sorted(params.keys())[-1] if param_key is None else param_key
    predictor.param_values = params.get(param_key)

    train, valid, test = datatools.load_mnist_npz(mnist_file)
    idx = np.random.permutation(len(valid[0]))[:2000]
    x_in = valid[0][idx]
    y_true = valid[1][idx]
    z_out = predictor(x_in=x_in)['z_out']

    imgfiles = [datatools.generate_imagename(i, y)
                for i, y in enumerate(idx, y_true)]

    labels = ['<img src="{}{}" width=100 height=100>'.format(URL_BASE, img)
              for img in imgfiles]

    palette = seaborn.color_palette("Set3", 10)
    colors = np.asarray([palette[y] for y in y_true])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    handle = ax.scatter(z_out.T[0], z_out.T[1],
                        c=colors, s=75, alpha=0.66)

    tooltip = plugins.PointHTMLTooltip(
        handle, labels,
        voffset=10, hoffset=10)

    plugins.connect(fig, tooltip)
    plt.show()
    if output_file:
        with open(output_file, 'w') as fp:
            mpld3.save_html(fig, fp)

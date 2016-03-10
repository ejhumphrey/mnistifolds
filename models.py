

import optimus


def pwrank(verbose=False):
    """Create a trainer and predictor graph for pairwise rank embeddings

    Returns
    -------
    trainer : optimus.Graph
        Graph with in-place updates for back-propagating error.

    predictor : optimus.Graph
        Static graph for processing images.
    """
    # 1.1 Create Inputs
    x_in = optimus.Input(
        name='x_in',
        shape=(None, 1, 28, 28))

    x_same = optimus.Input(
        name='x_same',
        shape=x_in.shape)

    x_diff = optimus.Input(
        name='x_diff',
        shape=x_in.shape)

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    margin_diff = optimus.Input(
        name='margin_diff',
        shape=None)

    margin_same = optimus.Input(
        name='margin_same',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=x_in.shape,
        weight_shape=(15, 1, 9, 9),
        pool_shape=(2, 2),
        act_type='relu')

    layer1 = optimus.Affine(
        name='layer1',
        input_shape=layer0.output.shape,
        output_shape=(None, 512,),
        act_type='tanh')

    layer2 = optimus.Affine(
        name='layer2',
        input_shape=layer1.output.shape,
        output_shape=(None, 2),
        act_type='linear')

    param_nodes = [layer0, layer1, layer2]
    # Create two copies
    nodes_same = [l.clone(l.name + "_same") for l in param_nodes]
    nodes_diff = [l.clone(l.name + "_diff") for l in param_nodes]

    # 1.1 Create Losses
    cost_sim = optimus.Euclidean(name='cost_sim')
    cost_diff = optimus.Euclidean(name='cost_diff')
    criterion = optimus.ContrastiveMargin(name='contrastive')

    loss_nodes = [cost_sim, cost_diff, criterion]

    # 1.2 Define outputs
    z_out = optimus.Output(name='z_out')
    loss = optimus.Output(name='loss')

    # 2. Define Edges
    base_edges = [
        (x_in, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, z_out)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (x_same, nodes_same[0].input),
            (nodes_same[0].output, nodes_same[1].input),
            (nodes_same[1].output, nodes_same[2].input),
            (x_diff, nodes_diff[0].input),
            (nodes_diff[0].output, nodes_diff[1].input),
            (nodes_diff[1].output, nodes_diff[2].input),
            (layer2.output, cost_sim.input_a),
            (nodes_same[2].output, cost_sim.input_b),
            (layer2.output, cost_diff.input_a),
            (nodes_diff[2].output, cost_diff.input_b),
            (cost_sim.output, criterion.cost_sim),
            (cost_diff.output, criterion.cost_diff),
            (margin_same, criterion.margin_sim),
            (margin_diff, criterion.margin_diff),
            (criterion.output, loss)])

    update_manager = optimus.ConnectionManager(
        list(map(lambda n: (learning_rate, n.weights), param_nodes)) +
        list(map(lambda n: (learning_rate, n.bias), param_nodes)))

    trainer = optimus.Graph(
        name='mnist_trainer',
        inputs=[x_in, x_same, x_diff,
                learning_rate, margin_same, margin_diff],
        nodes=param_nodes + nodes_same + nodes_diff + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss, z_out],
        loss=loss,
        updates=update_manager.connections,
        verbose=verbose)

    for node in param_nodes:
        optimus.random_init(node.weights, mean=0.0, std=0.1)

    predictor_edges = optimus.ConnectionManager(base_edges)

    predictor = optimus.Graph(
        name='mnist_embedding',
        inputs=[x_in],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[z_out],
        verbose=verbose)

    return trainer, predictor

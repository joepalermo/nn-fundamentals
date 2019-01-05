from utils import *



# build dataset
xs, ys = generate_spiral(1000, 2, 3, show=False)
train_xs, train_ys, test_xs, test_ys = train_test_split(xs, ys)

# normalize data
# train_mean = np.mean(train_xs, axis=0)
# train_std = np.std(train_xs, axis=0)
# train_xs = (train_xs - train_mean)/train_std

# specify neural network
layer_sizes = [2, 5, 3]
hyperparam_dict = {'mb_size': 2, 'lr': 0.0001}

# initialize data batching class (DataSet) and MLP
ds = DataSet(train_xs, train_ys, hyperparam_dict['mb_size'])
mlp = MLP(layer_sizes, hyperparam_dict)

n_epochs = 1
n_batches = n_epochs * ds.batches_in_epoch
for batch_i in range(5):
    mb_xs, mb_ys = ds.get_mb()
    mb_output = mlp.feedforward(mb_xs)
    print(mlp.loss(mb_output, mb_ys))
    dws, dbs = mlp.backprop(mb_output, mb_ys)
    # apply SGD
    for layer_i in range(mlp.n_layers):
        mlp.weights[layer_i] = mlp.weights[layer_i] - mlp.lr * dws[layer_i]
        mlp.biases[layer_i] = mlp.biases[layer_i] - mlp.lr * dbs[layer_i]

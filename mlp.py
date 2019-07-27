from utils import *

# build dataset
xs, ys = generate_spiral(1000, 2, 3, show=False)
train_xs, train_ys, test_xs, test_ys = train_test_split(xs, ys)

# normalize data
train_mean = np.mean(train_xs, axis=0)
train_std = np.std(train_xs, axis=0)
train_xs = (train_xs - train_mean)/train_std
test_xs = (test_xs - train_mean)/train_std

# specify neural network
layer_sizes = [2, 100, 3]
hyperparam_dict = {'mb_size': 64, 'lr': 0.001}

# initialize data batching class (DataSet) and MLP
ds = DataSet(train_xs, train_ys, hyperparam_dict['mb_size'])
mlp = MLP(layer_sizes, hyperparam_dict)

n_epochs = 1000
n_batches = n_epochs * ds.batches_in_epoch
for batch_i in range(n_batches):
    mb_xs, mb_ys = ds.get_mb()
    mlp.fit_mb(mb_xs, mb_ys)
    # test_output = mlp.feedforward(test_xs)
    # print(mlp.loss(test_output, test_ys))
test_output = mlp.feedforward(test_xs)
print(mlp.accuracy(test_output, test_ys))

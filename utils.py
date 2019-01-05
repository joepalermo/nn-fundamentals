import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

def generate_spiral(n, d, k, show=False):
    '''
    n number of points per class
    d dimensionality
    k number of classes
    '''
    xs = np.zeros((n*k,d)) # data matrix (each row = single example)
    ys = np.zeros(n*k, dtype='uint8') # class labels
    for j in range(k):
        ix = range(n*j,n*(j+1))
        r = np.linspace(0.0,1,n) # radius
        t = np.linspace(j*4,(j+1)*4,n) + np.random.randn(n)*0.2 # theta
        xs[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        ys[ix] = j
    if show:
        plt.scatter(xs[:, 0], xs[:, 1], c=ys, s=40, cmap=plt.cm.Spectral)
        plt.show()
    return xs, ys

def get_random_ints(n, k):
    '''get k random integers sampled without replacement from values 0 to n'''
    return np.random.choice(np.arange(n), k, replace=False)

def train_test_split(xs, ys, p_test=0.1, verbose=False):
    test_idxs =  get_random_ints(len(xs), int(len(xs)*p_test))
    test_xs = xs[test_idxs]
    train_idxs = np.array(list(set(np.arange(len(xs))) - set(test_idxs)))
    train_xs = xs[train_idxs]
    test_ys = ys[test_idxs]
    train_ys = ys[train_idxs]
    if verbose:
        print(train_xs.shape, train_ys.shape, test_xs.shape, test_ys.shape)
    return train_xs, train_ys, test_xs, test_ys

class DataSet:

    def __init__(self, xs, ys, mb_size, normalize=False):
        if normalize:
            self.xs = (xs - np.mean(xs, axis=0)) / np.stddev(xs, axis=0)
        else:
            self.xs = xs.copy()
        self.ys = ys.copy()
        self.mb_i = 0
        self.mb_size = mb_size
        self.batches_in_epoch = len(xs)//mb_size

    def get_mb(self, verbose=False):
        if verbose:
            print("generating mini-batch #", self.mb_i)
        if self.mb_i == 0:
            if verbose:
                print("shuffling the examples")
            idxs = np.arange(len(self.xs))
            np.random.shuffle(idxs)
            self.xs = self.xs[idxs]
            self.ys = self.ys[idxs]
        if (self.mb_i+1)*self.mb_size > len(self.xs):
            self.mb_i = 0
            return self.get_mb()
        mb_xs = self.xs[self.mb_i*self.mb_size:(self.mb_i+1)*self.mb_size]
        mb_ys = self.ys[self.mb_i*self.mb_size:(self.mb_i+1)*self.mb_size]
        self.mb_i += 1
        return mb_xs, mb_ys

# neural net utilities -----------------------

def print_shapes(ls):
    print([a.shape for a in ls])

def print_arrays(ls):
    for a in ls:
        print(a)

def relu(x):
    return np.clip(x, 0, np.inf)

def relu_grad(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])

def softmax_colwise(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def softmax(x):
    'softmax for row-wise examples'
    return np.exp(x)*(1/np.vstack([np.sum(np.exp(x),axis=1)]*x.shape[1]).T)
#
# def softmax(x):
#     'softmax for row-wise examples'
#     max_x = np.max(x, axis=1)
#     return np.exp(x-max_x)/np.vstack([np.sum(np.exp(x-max_x),axis=1)]*x.shape[1]).T

class MLP:

    def __init__(self, layer_sizes, hyperparam_dict):
        self.n_layers = len(layer_sizes) - 1
        self.wdims = [(layer_sizes[i],layer_sizes[i+1]) for i in range(self.n_layers)]
        self.weights = [np.sqrt(2/wd[0])*randn(wd[0], wd[1]) for wd in self.wdims]
        self.biases = [np.zeros(wd[1]) for wd in self.wdims]
        self.activations = [relu for _ in range(self.n_layers-1)] + [softmax]
        self.lr = hyperparam_dict['lr']

    def feedforward(self, xs, verbose=False):
        self.zs = list()
        self.post_zs = list()
        # the first post_z is just the input activation
        post_z = xs
        self.post_zs.append(post_z)
        for layer_i in range(self.n_layers):
            if verbose:
                print(f"layer {layer_i}: ")
                print(f"\tinput shape: {post_z.shape}")
                print(f"\tweight shape: {self.weights[layer_i].shape}")
                print(f"\tbias shape: {self.biases[layer_i].shape}")
            z = np.matmul(post_z, self.weights[layer_i]) + self.biases[layer_i]
            post_z = self.activations[layer_i](z)
            self.zs.append(z)
            self.post_zs.append(post_z)
        if verbose:
            print("zs")
            print_shapes(self.zs)
            print_arrays(self.zs)
            print("post_zs")
            print_shapes(self.post_zs)
            print_arrays(self.post_zs)
        return post_z

    def backprop(self, output, ys, verbose=False):
        dzs = list()
        row_idxs = list(range(len(output)))
        output[row_idxs, ys] -= 1 # convert output to the first dz
        dz = output
        dzs.append(dz)
        # compute all the dzs by backpropagating errors
        for layer_i in range(self.n_layers-1, 0, -1):
            da = np.dot(dz, self.weights[layer_i].T)
            dz = da * relu_grad(self.zs[layer_i-1]) # is this the right one?
            dzs.append(dz)
        dzs.append(None)
        # swap the order so that the first element is at shallowest depth
        dzs.reverse()
        # compute all the dws and dbs
        dws = list()
        dbs = list()
        for layer_i in range(self.n_layers):
            dw = np.dot(self.post_zs[layer_i].T, dzs[layer_i+1])
            db = np.sum(dzs[layer_i+1], axis=0)
            dws.append(dw)
            dbs.append(db)
        if verbose:
            print("dws: ")
            print_shapes(dws)
            print_arrays(dws)
            print("dbs: ")
            print_shapes(dbs)
            print_arrays(dbs)
        return dws, dbs

    def fit_mb(self, mb_xs, mb_ys):
        mb_output = self.feedforward(mb_xs)
        dws, dbs = self.backprop(mb_output, mb_ys)
        # apply SGD
        for layer_i in range(self.n_layers):
            self.weights[layer_i] = self.weights[layer_i] - self.lr * dws[layer_i]
            self.biases[layer_i] = self.biases[layer_i] - self.lr * dbs[layer_i]

    def loss(self, mb_output, mb_ys):
        return -np.mean(np.log(mb_output[np.arange(len(mb_output)), mb_ys]))

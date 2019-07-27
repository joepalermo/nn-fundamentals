import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.clip(x, 0, np.inf)


def relu_grad(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])


def softmax_colwise(x):
    'softmax for col-wise examples'
    max_x = np.max(x, axis=0, keepdims=True) # (1, n_examples)
    return np.exp(x-max_x)/np.sum(np.exp(x-max_x), axis=0)


def softmax(x):
    'softmax for row-wise examples'
    max_x = np.max(x, axis=1, keepdims=True) # (n_examples, 1)
    return (np.exp(x-max_x).T*(1/np.sum(np.exp(x-max_x), axis=1))).T


def print_shapes(ls):
    print([a.shape for a in ls])


def print_arrays(ls):
    for a in ls:
        print(a)


def get_random_ints(n, k):
    '''get k random integers sampled without replacement from values 0 to n'''
    return np.random.choice(np.arange(n), k, replace=False)


def generate_spiral(n, d, k, show=False):
    '''
    Taken from http://cs231n.github.io/
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


def train_test_split(xs, ys, p_test=0.1, verbose=False):
    test_idxs = get_random_ints(len(xs), int(len(xs)*p_test))
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
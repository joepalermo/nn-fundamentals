from utils import *


class MLP:

    def __init__(self, layer_sizes, hyperparam_dict):
        self.n_layers = len(layer_sizes) - 1
        self.wdims = [(layer_sizes[i],layer_sizes[i+1]) for i in range(self.n_layers)]
        self.weights = [np.sqrt(2/wd[0])*np.random.randn(wd[0], wd[1]) for wd in self.wdims]
        self.biases = [np.zeros(wd[1]) for wd in self.wdims]
        ## the following alternative initialization shows that you can optimize
        ## a neural network even if all weights have been initialized to 0
        ## but only if biases are initialized randomly
        # self.weights = [np.zeros((wd[0], wd[1])) for wd in self.wdims]
        # self.biases = [np.random.randn(wd[1]) for wd in self.wdims]
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

    def loss(self, output, ys):
        return -np.mean(np.log(output[np.arange(len(output)), ys]))

    def accuracy(self, output, ys):
        return 100*np.sum(np.argmax(output, axis=1) == ys)/len(output)

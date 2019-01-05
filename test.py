from utils import *

a = np.array([[ 1,  3,  4],
[ 2,  2,  1],
[ 9, -1,  7]])

print(np.round(softmax(a), 3))

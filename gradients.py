import math
import numpy as np

sig = lambda x: 1/(1+np.exp(-x)) # convenient function

## example 1
# x=-2;y=5;z=-4
#
# # building up the computation graph
# q = x+y
# f = q*z
#
# dq = z
# dz = q
# dx = dq * 1
# dy = dq * 1
#
# print(dx, dy, dz)

## example 2
# w = np.array([2,-3,-3]) # assume some random weights and data
# x = np.array([-1,-2,1])
#
# # forward pass
# dot = np.dot(w, x)
# f = sig(dot)
#
# # backward pass
# ddot = f * (1-f)
# dw = ddot * x
# print(dw)

## example 3
x = 3 # example values
y = -4

# forward pass
a = x + sig(y)
b = (x + y) ** 2
c = sig(x)
f = a / (b + c)

# backward pass
da = 1/(b+c)
db = -a/((b+c)**2)
dc = -a/((b+c)**2)
dx = dc * sig(x)*(1-sig(x))
dx += db * 2*(x+y)
dy = db * 2*(x+y)
dx += da * 1
dy += da * sig(y)*(1-sig(y))
print(dx, dy)

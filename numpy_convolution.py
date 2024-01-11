import numpy as np

def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(np.dot(x[i-1:i+2], w))
    return np.array(output)

x = np.arange(5)
w = np.array([2., 3., 4.])

result_convolve = convolve(x, w)
print(result_convolve)

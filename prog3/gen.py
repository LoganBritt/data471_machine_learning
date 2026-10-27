import numpy as np

#This is the example model to be generated in binary
W = np.array([
    [ 1.0,  0.0, -1.0],
    [ 0.0,  1.0, -1.0]
], dtype=np.float64)

b = np.array([0.0, 0.0, 0.0], dtype=np.float64)

np.savez("model.npz", W=W, b=b)

print("Test model saved to model.npz")

#This is the input to be generated
X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.0, 0.0]
], dtype=np.float64)

np.save("x.npy", X)

print("Test inputs saved to x.npy")


#This is the targets to be generated
y = np.array([0, 1, 0, 2], dtype=np.int64)

np.save("y.npy", y)

print("Test targets saved to y.npy")
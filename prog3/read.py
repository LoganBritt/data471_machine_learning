import sys
import numpy as np

print("Opening and reading file: " + sys.argv[1])
data = np.load(sys.argv[1])
print("Size of: " + str(np.shape(data)))
print(data)
import numpy as np
import random
from nn import NN
import gzip, os
np.set_printoptions(suppress=True)

# load the mnist dataset (from current directory)

def fetch(name):
  fp = os.path.join("", name)

  with open(fp, "rb") as f:
    dat = f.read()
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

X_train = fetch("../res/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28))
Y_train = fetch("../res/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("../res/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28))
Y_test = fetch("../res/t10k-labels-idx1-ubyte.gz")[8:]

print(np.asarray(X_train).shape)
print(np.asarray(Y_train).shape)
print(np.asarray(X_test).shape)
print(np.asarray(Y_test).shape)

random.shuffle(X_train)
random.shuffle(Y_train)
random.shuffle(X_test)
random.shuffle(Y_test)

# MAX = 3 

# X_train = X_train[:MAX]
# Y_train = Y_train[:MAX]
# X_test = X_test[:MAX]
# Y_test = Y_test[:MAX]

N = NN([784, 128, 10])
# N.train(X_train, Y_train)
N.train(X_test, Y_test)
print(N.test(X_test, Y_test))
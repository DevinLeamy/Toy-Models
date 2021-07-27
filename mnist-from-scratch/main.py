import numpy as np
import random
from nn import NN
# from comp import DeepNeuralNetwork
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

X_train = X_train / 255.0
X_test = X_test / 255.0

print(np.asarray(X_train).shape)
print(np.asarray(Y_train).shape)
print(np.asarray(X_test).shape)
print(np.asarray(Y_test).shape)

nn = NN([28 * 28, 128, 64, 10], epochs=1000)

nn.train(X_train, Y_train, X_test, Y_test)
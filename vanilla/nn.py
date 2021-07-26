from _math import Math
import time
import random
from tqdm import trange
import numpy as np
np.random.seed(0)
import time

class NN():
  epochs = 5
  batch_sz = 128

  def __init__(self, layers, learning_rate=0.001):
    self.layers = layers
    self.learning_rate = learning_rate 

    self.layer_cnt = len(layers)
    self.inputs = self.layers[0]
    self.outputs = self.layers[-1]

    self.params = self.initialize_parameters() 
 
  def initialize_parameters(self):
    IN = self.layers[0]
    H1 = self.layers[1]
    H2 = self.layers[2]
    OUT = self.layers[3]

    params = {
      'W1':np.random.randn(H1, IN) * np.sqrt(1. / H1),
      'W2':np.random.randn(H2, H1) * np.sqrt(1. / H2),
      'W3':np.random.randn(OUT, H2) * np.sqrt(1. / OUT)
    }

    return params

  
  def initialize_weight_matrix(self, in_nodes, out_nodes):
    return np.random.randn(out_nodes, in_nodes) * np.sqrt(1. / out_nodes)
  
  def forward(self, x): 
    params = self.params

    params["A0"] = x

    params["Z1"] = np.dot(params["W1"], params["A0"])
    params["A1"] = self.sigmoid(params["Z1"])

    params["Z2"] = np.dot(params["W2"], params["A1"])
    params["A2"] = self.sigmoid(params["Z2"])

    params["Z3"] = np.dot(params["W3"], params["A2"])
    params["A3"] = self.softmax(params["Z3"])

    # print(np.argmax(params["A3"]))
    # print(params["A3"])

    return params["A3"] 

  def sigmoid(self, x, derivative=False):
    if derivative:
      return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

  def softmax(self, x, derivative=False):
    exps = np.exp(x - x.max())
    if derivative:
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)
  
  def get_batch(self, inputs, outputs):
    return zip(inputs, outputs)
  
  def compute_accuracy(self, x_val, y_val):
    '''
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    '''
    predictions = []

    for x, y in zip(x_val, y_val):
      output = self.forward(x)
      pred = np.argmax(output)
      predictions.append(pred == np.argmax(y))
    
    return np.mean(predictions)

  def train(self, x_train, y_train, x_val, y_val):
    start_time = time.time()
    for iteration in range(self.epochs):
      for x,y in zip(x_train, y_train):
        output = self.forward(x)
        changes_to_w = self.backward(y, output)
        self.apply_gradients(changes_to_w)
      
      accuracy = self.compute_accuracy(x_val, y_val)
      print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
        iteration+1, time.time() - start_time, accuracy * 100
      ))
            
  
  def apply_gradients(self, grads):
    for key, value in grads.items():
      self.params[key] -= self.learning_rate * value
  
  def backward(self, desired_output, output):
    params = self.params
    grads = dict()

    error = 2 * (output - desired_output) / self.outputs * self.softmax(params["Z3"], derivative=True)
    grads["W3"] = np.outer(error, params["A2"]) 

    error = np.dot(params["W3"].T, error) * self.sigmoid(params["Z2"], derivative=True)
    grads["W2"] = np.outer(error, params["A1"])

    error = np.dot(params["W2"].T, error) * self.sigmoid(params["Z1"], derivative=True)
    grads["W1"] = np.outer(error, params["A0"])

    # print([activated_grads[key].shape for key in activated_grads.keys()])
    # print([np.sum(grads[key]) for key in grads.keys()])
    # print([grads[key].shape for key in grads.keys()])

    return grads
    
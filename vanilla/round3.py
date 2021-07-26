from _math import Math
import time
import random
from tqdm import trange
import numpy as np
np.random.seed(1)
import time

def sigmoid(x, derivative=False):
  if derivative:
    return sigmoid(x) * (1 - sigmoid(x)) 
  return 1 / (1 + np.exp(-x))

def softmax(x, derivative=False):
  if derivative:
    pass
  sum = np.sum(np.exp(x))
  return exp(x) / sum

class NN():
  def __init__(self, layers, epochs=1000, batch_sz=1000, l_rate=0.001):
    self.epochs = epochs
    self.batch_sz = batch_sz
    self.l_rate = l_rate
    self.IN, self.H1, self.H2, self.OUT = layers

    self.initialize_params()
  
  def initialize_params(self):
    self.l1 = self.initialize_weight_matrix(self.IN, self.H1)
    self.l2 = self.initialize_weight_matrix(self.H1, self.H2)
    self.l3 = self.initialize_weight_matrix(self.H2, self.OUT)
  
  def initialize_weight_matrix(self, c_layer, n_layer):
    return np.random.randn(n_layer, c_layer) * np.sqrt(1.0 / c_layer)
  
  def compute_unactivated(self, weights, previous_activated):
    return np.dot(weights, previous_activated)

  def compute_activated(self, unactivated):
    return sigmoid(unactivated)
  
  def process_output(self, output_layer):
    return np.argmax(output_layer)
  
  def update_weights(self, delta_l1, delta_l2, delta_l3):
    self.l1 -= delta_l1 * self.l_rate
    self.l2 -= delta_l2 * self.l_rate
    self.l3 -= delta_l3 * self.l_rate

  def forward_pass(self, x):
    cache = {} 
    cache["A0"] = x

    cache["Z1"] = self.compute_unactivated(self.l1, cache["A0"]) 
    cache["A1"] = self.compute_activated(cache["Z1"])

    cache["Z2"] = self.compute_unactivated(self.l2, cache["A1"]) 
    cache["A2"] = self.compute_activated(cache["Z2"])

    cache["Z3"] = self.compute_unactivated(self.l3, cache["A2"]) 
    cache["A3"] = self.compute_activated(cache["Z3"])
    # cache["A3"] = softmax(cache["Z3"])

    return cache
  
  def compute_derivative_c_wrt_a(self, y, y_hat):
    return 2.0 * (y_hat - y)
  
  def backward_pass(self, y, cache):
    grads = dict()
    y_hat = cache["A3"] 
    grads["A3"] = self.compute_derivative_c_wrt_a(y, y_hat)

    # partial = sigmoid(cache["Z3"], derivative=True) * grads["A3"] 
    partial = sigmoid(cache["Z3"], derivative=True) * grads["A3"] 
    grads["W3"] = np.outer(cache["A2"], partial)
    grads["A2"] = np.dot(self.l3.T, partial)

    partial = sigmoid(cache["Z2"], derivative=True) * grads["A2"]
    grads["W2"] = np.outer(cache["A1"], partial)
    grads["A1"] = np.dot(self.l2.T, partial)

    partial = sigmoid(cache["Z1"], derivative=True) * grads["A1"]
    grads["W1"] = np.outer(cache["A0"], partial)

    return (grads["W1"].T, grads["W2"].T, grads["W3"].T) 

  def format_output(self, y):
    formatted = np.zeros(self.OUT) 
    formatted[y] = 1.0
    return formatted
  
  def get_batch(self, inputs, outputs):
    indices = random.sample(range(len(inputs)), self.batch_sz)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]

    return zip(inputs, outputs)
  
  def test(self, inputs, outputs):
    guesses = []
    guesses_cache = [0 for _ in range(self.OUT)]
    for x, y in zip(inputs, outputs):
      y = self.format_output(y)

      cache = self.forward_pass(x)
      y = np.argmax(y)
      y_hat = self.process_output(cache["A3"])

      guesses.append(y == y_hat)
      guesses_cache[y_hat] += 1
    
    print(guesses_cache)
    return round(np.mean(guesses) * 100, 2)
  
  def train(self, train_x, train_y, test_x, test_y):
    accuracies = []
    for epoch in trange(self.epochs):
      t_delta_l1 = np.zeros(self.l1.shape) 
      t_delta_l2 = np.zeros(self.l2.shape) 
      t_delta_l3 = np.zeros(self.l3.shape) 
      for x, y in self.get_batch(train_x, train_y): 
        y = self.format_output(y)
        cache = self.forward_pass(x)
        delta_l1, delta_l2, delta_l3 = self.backward_pass(y, cache)

        # t_delta_l1 += delta_l1
        # t_delta_l2 += delta_l2
        # t_delta_l3 += delta_l3
        self.update_weights(delta_l1, delta_l2, delta_l3)
      # self.update_weights(t_delta_l1 / self.batch_sz, t_delta_l2 / self.batch_sz, t_delta_l3 / self.batch_sz)
      accuracies.append(self.test(test_x, test_y)))
      print(accuracies[-1])
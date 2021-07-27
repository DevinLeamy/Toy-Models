# from _math import Math
import time
import math
import random
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
import time

def sigmoid(x, derivative=False):
  if derivative:
    return sigmoid(x) * (1 - sigmoid(x)) 
  return 1 / (1 + np.exp(-x))

def softmax(x, derivative=False):
  if derivative:
    return softmax(x) * (1 - softmax(x))
  # (x - x.max()) - https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function/40576872
  exp = np.exp(x - x.max())
  return exp / np.sum(exp) 

def loss(y_hat, y):
  return np.sum((y_hat - y) ** 2)

class Record():
  def __init__(self, x, y, y_hat):
    self.x = x
    self.y = y
    self.y_hat = y_hat
  
  def guess_was_correct(self):
    return np.argmax(self.y) == np.argmax(self.y_hat)
  
  def compute_loss(self):
    return loss(self.y_hat, self.y)
    
class Plot():
  @staticmethod 
  def compute_epoch_accuracy(epoch_records):
    correct = [record.guess_was_correct() for record in epoch_records]
    return np.mean(correct)
  
  @staticmethod
  def compute_epoch_loss(epoch_records):
    losses = [record.compute_loss() for record in epoch_records]
    return np.mean(losses)

  @staticmethod
  def plot(epochs_records, loss=True, accuracy=True):
    losses = [Plot.compute_epoch_loss(epoch_records) for epoch_records in epochs_records] 
    accuracies = [Plot.compute_epoch_accuracy(epoch_records) for epoch_records in epochs_records]

    plt.figure(figsize=[7, 7])
    plt.ylim([0, 1.1])
    if loss:
      plt.plot(losses)
    if accuracy:
      plt.plot(accuracies)
    plt.show()
  
  @staticmethod
  def display_inputs(inputs):
    inputs = [input.reshape(28, 28) for input in inputs] 
    
    side_len = int(math.sqrt(len(inputs))) 
    plt.figure(figsize=[5, 5])

    number_rows = []
    for i in range(side_len):
      number_rows.append(np.hstack(inputs[i + j] for j in range(side_len)))
    
    number_matrix = np.vstack(row for row in number_rows)
    plt.imshow(number_matrix)
    plt.show()

class NN():
  def __init__(self, layers, epochs=10000, batch_sz=1000, l_rate=0.005):
    self.epochs = epochs
    self.batch_sz = batch_sz
    self.l_rate = l_rate
    self.IN, self.H1, self.H2, self.OUT = layers

    self.initialize_params()
  
  def initialize_params(self):
    self.l1 = dict()
    self.l2 = dict()
    self.l3 = dict()

    self.l1["W"] = self.initialize_weights(self.IN, self.H1)
    self.l1["B"] = self.initialize_bias(self.H1)

    self.l2["W"] = self.initialize_weights(self.H1, self.H2)
    self.l2["B"] = self.initialize_bias(self.H2)

    self.l3["W"] = self.initialize_weights(self.H2, self.OUT)
    self.l3["B"] = self.initialize_bias(self.OUT)
  
  def initialize_weights(self, c_layer, n_layer):
    return np.random.randn(n_layer, c_layer) * np.sqrt(1.0 / c_layer)
  
  def initialize_bias(self, n_layer):
    return np.random.randn(n_layer) * np.sqrt(1.0 / n_layer)
  
  def compute_unactivated(self, layer, previous_activated):
    return np.dot(layer["W"], previous_activated) + layer["B"]

  def compute_activated(self, unactivated):
    return sigmoid(unactivated)
  
  def process_output(self, output_layer):
    return np.argmax(output_layer)

  def update_weights(self, delta_l1, delta_l2, delta_l3):
    self.l1["W"] -= delta_l1["W"] * self.l_rate
    self.l1["B"] -= delta_l1["B"] * self.l_rate

    self.l2["W"] -= delta_l2["W"] * self.l_rate
    self.l2["B"] -= delta_l2["B"] * self.l_rate

    self.l3["W"] -= delta_l3["W"] * self.l_rate
    self.l3["B"] -= delta_l3["B"] * self.l_rate

  def forward_pass(self, x):
    cache = {} 
    cache["A0"] = x

    cache["Z1"] = self.compute_unactivated(self.l1, cache["A0"]) 
    cache["A1"] = self.compute_activated(cache["Z1"])

    cache["Z2"] = self.compute_unactivated(self.l2, cache["A1"]) 
    cache["A2"] = self.compute_activated(cache["Z2"])

    cache["Z3"] = self.compute_unactivated(self.l3, cache["A2"]) 
    cache["A3"] = softmax(cache["Z3"])

    return cache
  
  def compute_derivative_c_wrt_a(self, y, y_hat):
    return 2.0 * (y_hat - y)
  
  def backward_pass(self, y, cache):
    grads = dict()
    y_hat = cache["A3"] 
    grads["A3"] = self.compute_derivative_c_wrt_a(y, y_hat)

    partial = softmax(cache["Z3"], derivative=True) * grads["A3"] 
    grads["B3"] = partial
    grads["W3"] = np.outer(cache["A2"], partial)
    grads["A2"] = np.dot(self.l3["W"].T, partial)

    partial = sigmoid(cache["Z2"], derivative=True) * grads["A2"]
    grads["B2"] = partial
    grads["W2"] = np.outer(cache["A1"], partial)
    grads["A1"] = np.dot(self.l2["W"].T, partial)

    partial = sigmoid(cache["Z1"], derivative=True) * grads["A1"]
    grads["B1"] = partial
    grads["W1"] = np.outer(cache["A0"], partial)

    delta_l1 = { "W": grads["W1"].T, "B": grads["B1"] }
    delta_l2 = { "W": grads["W2"].T, "B": grads["B2"] }
    delta_l3 = { "W": grads["W3"].T, "B": grads["B3"] }

    return (delta_l1, delta_l2, delta_l3)

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
    correct = []
    guesses_cache = [0 for _ in range(self.OUT)]
    for x, y in zip(inputs, outputs):
      y = self.format_output(y)
      cache = self.forward_pass(x)

      y = np.argmax(y)
      y_hat = self.process_output(cache["A3"])

      correct.append(y == y_hat)
      guesses_cache[y_hat] += 1
    
    print(guesses_cache)
    return round(np.mean(correct) * 100, 2)
  
  def train(self, train_x, train_y, test_x, test_y):
    accuracies = []
    epochs_records = []
    for epoch in trange(self.epochs):
      epoch_records = []
      for x, y in self.get_batch(train_x, train_y): 
        y = self.format_output(y)
        cache = self.forward_pass(x)
        record = Record(x, y, cache["A3"])
        delta_l1, delta_l2, delta_l3 = self.backward_pass(y, cache)

        epoch_records.append(record)
        self.update_weights(delta_l1, delta_l2, delta_l3)

      epochs_records.append(epoch_records)
      accuracies.append(self.test(test_x, test_y))
      print(accuracies[-1])
    
    Plot.plot(epochs_records, loss=True, accuracy=True)
    Plot.display_inputs([record.x for record in epochs_records[-1] if not record.guess_was_correct()][:25])
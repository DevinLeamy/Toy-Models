from _math import Math
import random
from tqdm import trange
import numpy as np
import time

class NN:
  epochs = 10
  batch_sz = 128 

  def __init__(self, layers, learning_rate=0.001):
    self.layers = layers
    self.learning_rate = learning_rate 

    self.layer_cnt = len(layers)
    self.inputs = self.layers[0]
    self.outputs = self.layers[-1]

    self.weights = self.init_weights() 
    self.biases = self.init_biases() 
 
  def create_arr(self, len, random_init=True):
    if random_init:
      return [Math.rand() for _ in range(len)]
    return [0.0 for _ in range(len)]
  
  def create_matrix(self, rows, cols, random_init=True):
    return [self.create_arr(cols, random_init) for _ in range(rows)]

  def init_weights(self):
    weights = self.create_arr(self.layer_cnt) 
    '''
    Ex (3 x 3):
    [[w_11, w_12, w_13]
      [w_21, w_22, w_23],
      [w_31, w_32, w_33]]
    '''
    node_cnt = self.inputs
    for layer in range(self.layer_cnt):
      if layer == 0:
        continue
      nnode_cnt = self.layers[layer]
      weights[layer] = self.create_matrix(nnode_cnt, node_cnt) 

      node_cnt = nnode_cnt 
    return weights
  
  def init_biases(self):
    biases = self.create_arr(self.layer_cnt) 
    for layer in range(self.layer_cnt):
      if layer == 0:
        continue
      node_cnt = self.layers[layer]
      biases[layer] = self.create_arr(node_cnt)
    return biases
  
  def activation(self, x):
    return Math.sigmoid(x)
  
  def activate(self, arr):
    return [self.activation(x) for x in arr]
  
  import time
  def forward(self, x): 
    assert len(x) == self.inputs

    unactivated_layers = self.create_arr(self.layer_cnt, random_init=False)
    activated_layers = self.create_arr(self.layer_cnt, random_init=False)

    for layer in range(self.layer_cnt):
      start_time = time.time()
      # print("1 --- %.2f seconds ---" % (time.time() - start_time))
      if layer == 0:
        activated_layers[layer] = self.copy_shape(x, with_values=True)
        continue
      # print("2 --- %.2f seconds ---" % (time.time() - start_time))
      unactivated_layers[layer] = Math.dot(self.weights[layer], activated_layers[layer - 1])
      # print("3 --- %.2f seconds ---" % (time.time() - start_time))
      unactivated_layers[layer] = Math.add(self.biases[layer], unactivated_layers[layer])
      # print("4 --- %.2f seconds ---" % (time.time() - start_time))
      activated_layers[layer] = self.activate(unactivated_layers[layer]) 
      # print("5 --- %.2f seconds ---" % (time.time() - start_time))

    return (self.copy_shape(activated_layers[-1], with_values=True), unactivated_layers, activated_layers)
  
  def loss(self, result, desired): 
    assert len(result) == self.outputs

    return [Math.difference_squared(desired[i], result[i]) for i in range(self.outputs)]
  
  def format_output(self, output):
    if not isinstance(output, list):
      formatted = self.create_arr(self.outputs, random_init=False) 
      formatted[output] = 1.0
      return formatted
    return output
  
  def train(self, inputs, outputs):
    for epoch in trange(self.epochs):
      # random sample of indices for the batch
      batch_ids = random.sample(range(len(inputs)), self.batch_sz)

      total_weight_grads = self.copy_shape(self.weights, with_values=False)
      total_bias_grads = self.copy_shape(self.biases, with_values=False)

      for idx in batch_ids:
        input = inputs[idx]
        output = self.format_output(outputs[idx])

        _, unactivated_layers, activated_layers = self.forward(input)
        weight_grads, bias_grads = self.backward(output, unactivated_layers, activated_layers)

        total_weight_grads = Math.add(total_weight_grads, weight_grads)
        total_bias_grads = Math.add(total_bias_grads, bias_grads)
      
      average_gradient = lambda gradient: gradient / self.batch_sz
      averaged_weight_grads = Math.apply(total_weight_grads, average_gradient)
      averaged_bias_grads = Math.apply(total_bias_grads, average_gradient)

      self.apply_gradients(averaged_weight_grads, averaged_bias_grads)

    
  def parse_output(self, output):
    mx = max(output) 
    return output.index(mx)

  def test(self, inputs, outputs):
    n_tests = len(inputs)
    n_correct = 0
    for i in trange(n_tests):
      desired = outputs[i]
      result, _, _ = self.forward(inputs[i])
      guess = self.parse_output(result) 
      if guess == desired:
        n_correct += 1
    
    return n_correct / n_tests * 100
  
  def copy_shape(self, data, with_values=False):
    if not isinstance(data, list):
      return data if with_values else 0.0 
    return [self.copy_shape(row) for row in data]
  
  def get_bias_grad(self, derivate_activated_wrt_unactived, derivate_activated_wrt_cost):
    return derivate_activated_wrt_unactived * derivate_activated_wrt_cost
  
  def get_activated_grad(self, result, desired):
    return Math.derivative_difference_squared(result, desired)
  
  def get_weight_grad(self, derivative_unactivated_wrt_weight, derivate_activated_wrt_unactived, derivated_activated_wrt_cost):
    return derivative_unactivated_wrt_weight * derivate_activated_wrt_unactived * derivated_activated_wrt_cost 
  
  def apply_gradients(self, weight_grads, bias_grads):
    apply_gradients = lambda previous_value, gradient: previous_value - gradient * self.learning_rate
    self.weights = Math.component_wise_apply(self.weights, weight_grads, apply_gradients)
    self.biases = Math.component_wise_apply(self.biases, bias_grads, apply_gradients)

  def backward(self, desired_output, unactivated_layers, activated_layers):
    weight_grads = self.copy_shape(self.weights)
    activated_grads = self.copy_shape(self.biases)
    activated_grads[0] = self.create_arr(self.inputs, random_init=False)
    bias_grads = self.copy_shape(self.biases)

    # print(np.asarray(a_layers).shape)
    
    for i in range(self.outputs):
      result = activated_layers[-1][i]
      desired = desired_output[i]
      activated_grads[-1][i] = self.get_activated_grad(result, desired) 
    
    # calculate gradients
    for layer in reversed(range(self.layer_cnt)):
      if layer == 0:
        continue

      for dest in range(self.layers[layer]):
        unactivated_output = unactivated_layers[layer][dest]
        d_activated_wrt_unactivated = Math.derivative_sigmoid(unactivated_output)
        bias_grads[layer][dest] = self.get_bias_grad(d_activated_wrt_unactivated, activated_grads[layer][dest])
 
        prev_layer = layer - 1
        prev_layer_n_cnt = self.layers[prev_layer]

        for src in range(prev_layer_n_cnt):
          activated_output = activated_layers[prev_layer][src]
          d_unactivated_wrt_weight = activated_output

          weight_grads[layer][dest][src] = self.get_weight_grad(d_unactivated_wrt_weight, d_activated_wrt_unactivated, activated_grads[layer][dest])

          if prev_layer >= 0: 
            weight = self.weights[layer][dest][src]
            d_unactivated_wrt_activated = weight
            activated_grads[prev_layer][src] += d_unactivated_wrt_activated * d_activated_wrt_unactivated * activated_grads[layer][dest]
    
    return (weight_grads, bias_grads)
    
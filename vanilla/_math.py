import math
import random
import numpy as np
random.seed(1)

class Math:
  @staticmethod 
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  @staticmethod
  def softmax(x, derivative=False):
    exps = np.exp(x - x.max())
    if derivative:
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

  @staticmethod
  def difference_squared(a, b):
    return pow(a - b, 2)
  
  @staticmethod
  def derivative_difference_squared(a, b):
    return 2 * (a -  b)
  
  @staticmethod
  def derivative_sigmoid(x): 
    # return (np.exp(-x))/((np.exp(-x)+1)**2)
    return Math.sigmoid(x) * (1 - Math.sigmoid(x))
import math
import random
import numpy as np
random.seed(1)

class Math:
  @staticmethod 
  def dot(m, v):
    return np.asarray(m).dot(np.asarray(v))
  
  @staticmethod
  def component_wise_apply(m1, m2, fn):
    if isinstance(m1, list):
      assert len(m1) == len(m2)
      return [Math.component_wise_apply(row1, row2, fn) for row1, row2 in zip(m1, m2)]
    return fn(m1, m2)

  @staticmethod
  def apply(m, fn):
    if isinstance(m, list):
      return [Math.apply(row, fn) for row in m]
    return fn(m)

  @staticmethod
  def add(m1, m2):
    if isinstance(m1, list) and isinstance(m2, list):
      assert len(m1) == len(m2)
      return [Math.add(row1, row2) for row1, row2 in zip(m1, m2)]
    return np.asarray(m1) + np.asarray(m2)
  
  @staticmethod 
  def sigmoid(x):
    try:
      return 1 / (1 + math.exp(-x))
    except:
      return 0.0
  
  @staticmethod
  def difference_squared(a, b):
    return pow(a - b, 2)
  
  @staticmethod
  def derivative_difference_squared(a, b):
    return 2 * (a -  b)
  
  @staticmethod
  def derivative_sigmoid(x): # derivative of sigmoid
    return Math.sigmoid(x) * (1 - Math.sigmoid(x))
  
  @staticmethod
  def rand():
    return random.uniform(-1, 1)

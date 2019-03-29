import tensorflow as tf

class Structure():
  def __init__(self, layer_list):
    print("Creating structure for network using " + layer_list)

class Layer():
    def __init__(self, act_function=sigmoid):
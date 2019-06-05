import tensorflow as tf
import scipy.io
import numpy as np
import argparse
import time
import settings

class dense_layers():
    def __init__(self,A,a):
      self.weights = tf.constant(A)
      self.bias = tf.constant(a)

class conv_layers():
    def __init__(self,A,a,stride):
        self.weights = tf.constant(A)
        if np.shape(a) != ():
          self.bias = tf.constant(a)
        else:
          self.bias = tf.zeros([self.weights.shape[-1]])
        self.stride = stride
      
def forward_pass(x,bs,hps):
  conv_l, dense_l = settings.layers[0], settings.layers[1]
  if hps.dataset == 'cifar10':
     if hps.model == 'l2-at':
       y = x - tf.ones([bs,32,32,3],dtype=tf.float32)*0.5
     else:
       y = x - 0.0
       
     y = tf.nn.relu(tf.nn.conv2d(y, conv_l[0].weights, strides=[1, conv_l[0].stride, conv_l[0].stride, 1], padding="SAME") - conv_l[0].bias)
     for counter in range(1,len(conv_l)):
       y = tf.nn.relu(tf.nn.conv2d(y, conv_l[counter].weights, strides=[1, conv_l[counter].stride, conv_l[counter].stride, 1], padding="SAME") - conv_l[counter].bias)
     y = tf.squeeze(y)
     y = tf.reshape(y,[bs,-1])
     for counter in range(0,len(dense_l)-1):
       y = tf.nn.relu(tf.matmul(y,dense_l[counter].weights) - dense_l[counter].bias)
     return tf.matmul(y,dense_l[-1].weights) - dense_l[-1].bias
     
  elif hps.dataset == 'mnist':
     maxp_size = [2, 2]
     maxp_stride = [2, 2]
     y = tf.nn.relu(tf.nn.conv2d(x, conv_l[0].weights, strides=[1, conv_l[0].stride, conv_l[0].stride, 1], padding="SAME") - conv_l[0].bias)
     y = tf.nn.max_pool(y,[1,maxp_size[0],maxp_size[0],1],[1,maxp_stride[0],maxp_stride[0],1],padding="VALID")
     for counter in range(1,len(conv_l)):
       y = tf.nn.relu(tf.nn.conv2d(y, conv_l[counter].weights, strides=[1, conv_l[counter].stride, conv_l[counter].stride, 1], padding="SAME") - conv_l[counter].bias)
       y = tf.nn.max_pool(y,[1,maxp_size[1],maxp_size[1],1],[1,maxp_stride[1],maxp_stride[1],1],padding="VALID")
     y = tf.squeeze(y)
     y = tf.reshape(y,[bs,-1])
     for counter in range(0,len(dense_l)-1):
       y = tf.nn.relu(tf.matmul(y,dense_l[counter].weights) - dense_l[counter].bias)
     return tf.matmul(y,dense_l[-1].weights) - dense_l[-1].bias
    
def get_weights_conv(model, hps):
  if hps.dataset == 'cifar10':
    stride = [1,1,2,1,1,2,1,2]
    if hps.model == 'plain':
      conv_l = [conv_layers(model['A0'],model['bA0'],stride[0]), conv_layers(model['A1'],model['bA1'],stride[1]), conv_layers(model['A2'],model['bA2'],stride[2]), conv_layers(model['A3'],model['bA3'],stride[3]), conv_layers(model['A4'],model['bA4'],stride[4]), conv_layers(model['A5'],model['bA5'],stride[5]), conv_layers(model['A6'],model['bA6'],stride[6]), conv_layers(model['A7'],model['bA7'],stride[7])]
      dense_l = [dense_layers(model['A8'],model['bA8']), dense_layers(model['A9'],model['bA9'])]
    elif hps.model == 'linf-at' or hps.model == 'l2-at': 
      conv_l = [conv_layers(model['A0'],0,stride[0]), conv_layers(model['A1'],0,stride[1]), conv_layers(model['A2'],0,stride[2]), conv_layers(model['A3'],0,stride[3]), conv_layers(model['A4'],0,stride[4]), conv_layers(model['A5'],0,stride[5]), conv_layers(model['A6'],0,stride[6]), conv_layers(model['A7'],0,stride[7])]
      dense_l = [dense_layers(model['A8'],-model['A9']), dense_layers(model['A10'],-model['A11'])]
    else:
      raise ValueError('unknown model')
  
  elif hps.dataset == 'mnist':
    conv_l = [conv_layers(model['A0'],-model['A1'],1), conv_layers(model['A2'],-model['A3'],1)]
    dense_l = [dense_layers(model['A4'],-model['A5']), dense_layers(model['A6'],-model['A7'])]
     
  return conv_l,dense_l
    
class Model():
  def __init__(self, hps):
    self._build_model(hps)
    
  def _build_model(self, hps):
      if hps.dataset in ['cifar10']:
        self.x_input = tf.placeholder(
          tf.float32,
          shape=[None, 32, 32, 3])
      elif hps.dataset == 'mnist':
        self.x_input = tf.placeholder(
          tf.float32,
          shape=[None, 28, 28, 1])
          
      self.hps = hps

      self.y_input = tf.placeholder(tf.int64, shape=None)
      
      self.bs = tf.placeholder(tf.int32, shape=None)
      
      self.y = forward_pass(self.x_input, self.bs, self.hps)
      
      self.predictions = tf.argmax(self.y, 1)
      self.correct_prediction = tf.equal(self.predictions, tf.squeeze(self.y_input))
      self.corr_pred = self.correct_prediction
      self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
  
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.grad = tf.gradients(self.xent, self.x_input)[0]
      
      print('model built')
    
def load_dataset(hps):
  if hps.dataset == 'cifar10':
    cifar10_data = scipy.io.loadmat("datasets/cifar10/cifar10_test.mat")
    x_test = cifar10_data['Xtest']
    y_test = cifar10_data['Ytest']
    
    x_test.astype(np.float32)
    y_test_0 = y_test
    y_test = np.eye(10)[y_test]
    y_test=np.squeeze(y_test)
  
    stride = [1,1,2,1,1,2,1,2]
    
    if hps.model == 'plain': model = scipy.io.loadmat("models/cifar10_weights_plain.mat")
    elif hps.model == 'linf-at': model = scipy.io.loadmat("models/cifar10_weights_linf.mat")
    elif hps.model == 'l2-at': model = scipy.io.loadmat("models/cifar10_weights_l2.mat")
    else: raise ValueError('unknown model')
     
  elif hps.dataset == 'mnist':
    
    mnist_data = scipy.io.loadmat("datasets/mnist/mnist_test.mat")
    x_test = mnist_data['X_test']
    x_test.astype(np.float32)
    x_test = np.expand_dims(x_test,3)
    y_test = mnist_data['label_test']
    x_test.astype(np.float32)
    y_test_0 = y_test
    y_test = np.eye(10)[y_test]
    y_test=np.squeeze(y_test)
    
    maxp_size = [2,2]
    maxp_stride = [2,2]
    
    if hps.model == 'plain': model = scipy.io.loadmat("models/mnist_weights_plain.mat")
    elif hps.model == 'linf-at': model = scipy.io.loadmat("models/mnist_weights_linf.mat")
    elif hps.model == 'l2-at': model = scipy.io.loadmat("models/mnist_weights_l2.mat")
    else: raise ValueError('unknown model')
     
  return model, x_test, y_test, y_test_0
     
  


import pickle as cPickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('C:/Users/rgupta5/Downloads/mnist.pkl.gz', 'rb')
    u = cPickle._Unpickler( f )
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

import random

class Network(object):
	def __init__(self,layers): 
		self.num_layers = len(layers)  
		self.layers = layers
		self.biases = [np.random.randn(y,1) for y in layers[1:]]
		self.weights = [np.random.randn(y,x) 
						for x,y in zip(layers[:-1],layers[1:])]

	def feedforward(self,a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)		
		return a
		
	def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size,eta,test_data = None):
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)       
			mini_batches = [ training_data[k:k + mini_batch_size]
			                               for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print("Epoch {0}: {1}%".format(j,self.evaluate(test_data)/n_test*100))
			else:
				print("Epoch {0} complete".format(j))
			
	def update_mini_batch(self,mini_batch,eta):
		bb = [np.zeros(b.shape) for b in self.biases]
		ww = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_bb, delta_ww = self.backprop(x, y)
			bb = [nb+dnb for nb, dnb in zip(bb, delta_bb)]
			ww = [nw+dnw for nw, dnw in zip(ww, delta_ww)]
		self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, ww)]
		self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, bb)]
	
	def backprop(self, x, y):
		bb = [np.zeros(b.shape) for b in self.biases]
		ww = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x] 
		zs = [] 
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		bb[-1] = delta
		ww[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			bb[-l] = delta
			ww[-l] = np.dot(delta, activations[-l-1].transpose())
		return (bb, ww)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

import sys
sys.path.append("../")

training_data, validation_data, test_data = load_data_wrapper()
net = Network([784, 30, 10])
net.stochastic_gradient_descent(training_data, 33, 10, 5.0, test_data=test_data)

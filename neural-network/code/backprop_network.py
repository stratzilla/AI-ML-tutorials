#!/usr/bin/env python3

import random # for shuffling data and weight init 
import csv # for reading data file
import matplotlib.pyplot as plt # for plotting
from math import exp, floor, ceil

def stochastic_gradient_descent(network, classes, training_data):
	"""Training function for neural network.
	Performs feed-forward, backpropagation, and update weight functions.

	Parameters:
		network : the neural network.
		classes : the number of classes for the data.
		training_data : data to train the network on.
	"""
	for _ in range(0, EPOCHS):
		# there is no temporal delta and therefore no momentum for the first
		# training example, so skip that step later for first example
		first_example = True
		total_error = 0.00
		for example in training_data:
			# skip if not first example; keep track of prior example delta
			temporal_delta = [neuron['d'] \
				for layer in network for neuron in layer] \
				if not first_example else None
			# create a list of possible outputs
			outputs = [0 for _ in range(classes)]
			outputs[int(example[-1])] = 1 # denote correct classification
			# get actual output from feed forward pass. Feeding forward will
			# also initialize network neuron outputs to be used in backprop
			actual = feed_forward(network, example)
			total_error += sse(actual, outputs) # aggregate error
			# perform backpropagation to propagate error through network
			backpropagate(network, outputs)
			# update weights based on network params and neuron contents
			update_weights(network, example, temporal_delta)
			# to not clobber neural outputs, reset them to zero
			reset_neurons(network)
			first_example = False # now we can consider momentum
		# append results for this epoch to global lists to make plots
		MSE.append(total_error/len(training_data))
		TRP.append(performance_measure(NETWORK, TRAIN))
		TEP.append(performance_measure(NETWORK, TEST))

def feed_forward(network, example):
	"""Feedforward method. Feeds data forward through network.

	Parameters:
		network : the neural network.
		example : an example of data to feed forward.

	Returns:
		The output of the forward pass.
	"""
	layer_input, layer_output = example, []
	for layer in network:
		for neuron in layer:
			# sum the weight with inputs
			summ = summing_function(neuron['w'], layer_input)
			# activate the sum, store output
			neuron['o'] = activation_function(summ) 
			# append output to outputs
			layer_output.append(neuron['o']) 
		# inputs become outputs of previous layer
		layer_input, layer_output = layer_output, [] 
	return layer_input

def backpropagate(network, example):
	"""Backpropagation function. Backpropagates error through network.

	Parameters:
		network : the neural network.
		example : a training example.
	"""
	for i in range(len(network)-1, -1, -1): # for each reverse order layer
		for j in range(len(network[i])): # for each neuron in the layer
			err = 0.00
			if i == len(network)-1: # if the output layer
				# error is a function of what the output is versus the target
				err = example[j] - network[i][j]['o']
			else: # if an inner layer
				summ = 0.00
				# error is the sum of neuron weights times their deltas
				for neuron in network[i+1]: 
					summ += neuron['w'][j] * neuron['d']
				err = summ
			# delta is amount of correction
			network[i][j]['d'] = activation_derivative(network[i][j]['o']) * err

def reset_neurons(network):
	"""Resets neural outputs to zero after each backprop pass.

	Parameters:
		network : the neural network.
	"""
	for layer in network:
		for neuron in layer:
			# we don't want to clobber the outputs later
			# so reset after every example
			neuron['o'] = 0

def update_weights(network, example, delta):
	"""Function to update network weights.

	Parameters:
		network : the neural network.
		example : a training example.
		delta : temporal delta.
	"""
	for i in range(len(network)): # for each layer in network
		# we update the weights in order of layers 0..n
		# so we need either the example or the outputs of a layer
		if i != 0: # if not first layer
			# init t as neural outputs
			t = [neuron['o'] for neuron in network[i-1]]
		else: # if first layer
			t = example[:-1] # init t as the training example attributes
		# for each neuron in layer; zip with a length of network variable 
		# to access deltas
		for neuron, d in zip(network[i], range(0, len(network[i]))):
			for f in range(len(t)): # for each feature or output of t
				# update weight based on learning rate term
				neuron['w'][f] += LEARNING_RATE * float(t[f]) * neuron['d']
				if delta is not None: # if there is a temporal delta
					# update weight based on momentum rate term
					neuron['w'][f] += MOMENTUM_RATE * delta[d]
				# also update neural bias
				neuron['w'][-1] += LEARNING_RATE * neuron['d']

def sse(actual, target):
	"""Sum Square Error loss function.
	Determines error of network.

	Parameters:
		actual : the actual output from the network.
		target : the expected output from the network.
		
	Returns:
		The sum squared error of the network.
	"""
	summ = 0.00
	for i in range(len(actual)):
		summ += (actual[i] - target[i])**2
	return summ

def activation_function(z):
	"""
	Parameters:
		z : summed output of neuron.

	Returns:
		The neuron activation based on the summed output.
	"""
	return 1 / (1 + exp(-z))

def activation_derivative(z):
	"""Derivative of logarithmic sigmoid function.

	Parameters:
		z : summing output.

	Returns:
		The differential of the neural output.
	"""
	return z * (1 - z)

def summing_function(weights, inputs):
	"""Sums the synapse weights with inputs and bias.

	Parameters:
		weights : synaptic weights.
		inputs : a vector of inputs.

	Returns:
		The aggregate of inputs times weights, plus bias.
	"""
	bias = weights[-1] # bias is the final value in the weight vector
	summ = 0.00 # to sum
	for i in range(len(weights)-1):
		# aggregate the weights with input values
		summ += (weights[i] * float(inputs[i]))
	return summ + bias

def performance_measure(network, data):
	"""Measures accuracy of the network using classification error.

	Parameters:
		network : the neural network.
		data : a set of data examples.

	Returns:
		A percentage of correct classifications.
	"""
	correct, total = 0, 0
	for example in data:
		# check to see if the network output matches target output
		if check_output(network, example) == float(example[-1]):
			correct += 1
		total += 1
	return 100*(correct / total)

def check_output(network, example):
	"""Compares network output to actual output.

	Parameters:
		network : the neural network.
		example : an example of data.

	Returns:
		The class the example belongs to (based on network guess).
	"""
	output = feed_forward(network, example)
	return output.index(max(output))

def initialize_network(n, h, o):
	"""Neural network initializer.
	The network will be structured as nested data structures, namely a list of
	lists of dicts. As the algorithm continues, not only the weights will be
	stored but also deltas, outputs, errors.

	Parameters:
		n : the number of input neurons.
		h : the number of hidden neurons.
		o : the number of output neurons.

	Returns:
		An n-h-o neural network as a list of list of dicts.
	"""
	def r(): # an inline function to generate randomly uniform numbers
		return random.uniform(-0.50, 0.50)
	neural_network = [] # initially an empty list
	# there are (n * h) connections between input layer and hidden layer
	# a 'w' will denote weights
	neural_network.append([{'w':[r() for i in range(n+1)]} for j in range(h)])
	# there are (h * o) connections between hidden layer and output layer
	neural_network.append([{'w':[r() for i in range(h+1)]} for j in range(o)])
	return neural_network

def load_data(filename):
	"""Loads CSV for splitting into training and testing data.

	Parameters:
		filename : the filename of the file to load.

	Returns:
		Two lists, each corresponding to training and testing data.
	"""
	with open(filename, newline='\n') as csv_file: # open CSV
		data = [] # store data as a master list
		rows = csv.reader(csv_file, delimiter=',') # get each row
		for row in rows: # for each row
			data.append(row) # add to data list
	random.shuffle(data) # randomize data order
	training_data = data[:floor(len(data)*0.70)] # 70% split train
	testing_data = data[-ceil(len(data)*0.30):] # 30% split test
	return training_data, testing_data

def plot_data():
	"""Plots data.
	Displays MSE, training accuracy, and testing accuracy over time.
	"""
	x = range(0, EPOCHS)
	fig, ax2 = plt.subplots()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('MSE', color='blue')
	line, = ax2.plot(x, MSE, '-', c='blue', lw='1', label='MSE')
	ax1 = ax2.twinx()
	ax1.set_ylabel('Accuracy', color='green')
	line2, = ax1.plot(x, TRP, '-', c='green', lw='1', label='Training')
	line3, = ax1.plot(x, TEP, ':', c='green', lw='1', label='Testing')
	fig.tight_layout()
	fig.legend(loc='center')
	plt.show()
	plt.clf()

if __name__ == '__main__':
	TRAIN, TEST = load_data('../data/iris.csv')
	FEATURES = len(TRAIN[0][:-1])
	CLASSES = len(list(set([c[-1] for c in TRAIN])))
	NETWORK = initialize_network(FEATURES, 5, CLASSES)
	LEARNING_RATE, MOMENTUM_RATE = 0.100, 0.001
	EPOCHS = 250
	MSE, TRP, TEP = [], [], []
	stochastic_gradient_descent(NETWORK, CLASSES, TRAIN)
	plot_data()

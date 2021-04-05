#!/usr/bin/env python3

import pandas as pd
import random
import matplotlib.pyplot as plt

class Particle:
	def __init__(self, pos, vel):
		self.pos, self.vel = pos, vel
		network = initialize_network(self.pos)
		self.fit = mse(network)
		self.best_pos, self.best_fit = self.pos, self.fit
	
	def set_pos(self, pos):
		self.pos = pos
		if not any(p < -7.00 for p in pos)\
		and not any(p > 7.00 for p in pos):
			network = initialize_network(self.pos)
			fitness = mse(network)
			if fitness < self.best_fit:
				self.fit = fitness
				self.best_fit = self.fit
				self.best_pos = self.pos
	
	def set_vel(self, vel):
		self.vel = vel
	
	def get_pos(self):
		return self.pos
	
	def get_vel(self):
		return self.vel
	
	def get_best_pos(self):
		return self.best_pos

	def get_fit(self):
		return self.fit

def pso(dim, epochs, swarm_size, ic, cc, sc):
	swarm = initialize_swarm(swarm_size, dim)
	for e in range(1, epochs+1):
		swarm_best = get_swarm_best(swarm)
		MSE.append(swarm_best[0])
		TRP.append(performance_measure(swarm_best[1], TRAIN))
		TEP.append(performance_measure(swarm_best[1], TEST))
		move_particles(swarm, dim, ic, cc, sc)
	return get_swarm_best(swarm)[0]

def move_particles(swarm, dim, ic, cc, sc):
	swarm_best = get_swarm_best(swarm)
	for particle in swarm:
		new_pos = [0 for _ in range(dim)]
		new_vel = [0 for _ in range(dim)]
		for d in range(dim):
			r_1 = random.uniform(0.00, 1.00)
			r_2 = random.uniform(0.00, 1.00)
			weight = ic * particle.get_vel()[d]
			cognitive = cc * r_1
			cognitive *= (particle.get_best_pos()[d] - particle.get_pos()[d])
			social = sc * r_2
			social *= (swarm_best[1][d] - particle.get_pos()[d])
			new_vel[d] = weight + cognitive + social
			new_pos[d] = particle.get_pos()[d] + new_vel[d]
		particle.set_pos(new_pos)
		particle.set_vel(new_vel)

def feed_forward(network, example):
	layer_input, layer_output = example, []
	for layer in network:
		for neuron in layer:
			summ = summing_function(neuron, layer_input)
			layer_output.append(activation_function(summ))
		layer_input, layer_output = layer_output, []
	return layer_input

def summing_function(weights, inputs):
	bias = weights[-1]
	summ = 0.00
	for i in range(len(weights)-1):
		summ += (weights[i] * float(inputs[i]))
	return summ + bias

def activation_function(z):
	return z if z >= 0 else 0

def sse(actual, target):
	summ = 0.00
	for i in range(len(actual)):
		summ += (actual[i] - target[i])**2
	return summ

def mse(network):
	training = TRAIN
	summ = 0.00
	for example in training:
		target = [0 for _ in range(CLASSES)]
		target[int(example[-1])] = 1
		actual = feed_forward(network, example)
		summ += sse(actual, target)
	return summ / len(training)

def performance_measure(particle, data):
	network = initialize_network(particle)
	correct, total = 0, 0
	for example in data:
		if check_output(network, example) == float(example[-1]):
			correct += 1
		total += 1
	return 100*(correct / total)

def check_output(network, example):
	output = feed_forward(network, example)
	return output.index(max(output))

def initialize_network(p):
	n, h, o = FEATURES, HIDDEN_SIZE, CLASSES
	part = iter(p)
	neural_network = []
	neural_network.append([[next(part) for i in range(n+1)] for j in range(h)])
	neural_network.append([[next(part) for i in range(h+1)] for j in range(o)])
	return neural_network

def initialize_swarm(size, dim):
	swarm = []
	for _ in range(size):
		position = [random.uniform(-1.00, 1.00) for _ in range(dim)]
		velocity = [0 for _ in range(dim)]
		particle = Particle(position, velocity)
		swarm.append(particle)
	return swarm

def get_swarm_best(swarm):
	best_fit = swarm[0].get_fit()
	best_pos = swarm[0].get_pos()
	for particle in swarm:
		if particle.get_fit() < best_fit:
			best_fit = particle.get_fit()
			best_pos = particle.get_pos()
	return best_fit, best_pos

def load_data(filename):
	df = pd.read_csv(filename, header=None, dtype=float)
	for features in range(len(df.columns)-1):
		df[features] = (df[features] - df[features].mean())/df[features].std()
	train = df.sample(frac=0.70).fillna(0.00)
	test = df.drop(train.index).fillna(0.00)
	return train.values.tolist(), test.values.tolist()

def plot_data():
	x = range(0, EPOCHS)
	fig, ax2 = plt.subplots()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('MSE', color='blue')
	line, = ax2.plot(x, MSE, '-', c='blue', lw='1', label='MSE')
	ax1 = ax2.twinx()
	ax1.set_ylabel('Accuracy (%)', color='green')
	line2, = ax1.plot(x, TRP, '-', c='green', lw='1', label='Training')
	line3, = ax1.plot(x, TEP, ':', c='green', lw='1', label='Testing')
	fig.tight_layout()
	fig.legend(loc='center')
	ax1.set_ylim(0, 101)
	plt.show()
	plt.clf()
	
if __name__ == '__main__':
	MSE = []
	TRP = []
	TEP = []
	W = 0.3
	C_1 = 1.5
	C_2 = 1.2
	TRAIN, TEST = load_data('../data/wine.csv')
	FEATURES = len(TRAIN[0][:-1])
	CLASSES = len(list(set([c[-1] for c in (TRAIN+TEST)])))
	HIDDEN_SIZE = 5
	DIMENSIONS = (HIDDEN_SIZE * (FEATURES+1)) + \
		(CLASSES * (HIDDEN_SIZE+1))
	SWARM_SIZE = 100
	EPOCHS = 100
	NETWORK = pso(DIMENSIONS, EPOCHS, SWARM_SIZE, W, C_1, C_2)
	plot_data()
#!/usr/bin/env python3

import pandas as pd
from math import floor, ceil, exp
import matplotlib.pyplot as plt
import random

class Chromosome:
	def __init__(self, genes, fit=None):
		self.genes = genes
		if fit is None:
			network = initialize_network(self.genes)
			self.fit = mse(network)
		else:
			self.fit = fit
	
	def set_genes(self, genes):
		self.genes = genes
		network = initialize_network(self.genes)
		self.fit = mse(network)
	
	def get_genes(self):
		return self.genes
	
	def get_fit(self):
		return self.fit

	def __lt__(self, other):
		return self.fit < other.fit
	
	def __getitem__(self, key):
		return self.genes[key]

def genetic_network(el_p, to_p, dim, epochs, pop_size, cr, mr):
	population = initialize_population(pop_size, dim)
	for e in range(1, epochs+1):
		population.sort()
		MSE.append(population[0].get_fit())
		TRP.append(performance_measure(population[0].get_genes(), TRAIN))
		TEP.append(performance_measure(population[0].get_genes(), TEST))
		mating_pool = []
		elites = elite_selection(population, el_p)
		del population[:len(elites)]
		t_winner = tournament_selection(population, to_p)
		mating_pool.extend(elites)
		mating_pool.append(t_winner)
		population = evolve(mating_pool, elites, pop_size, cr, mr)
		mating_pool.clear()
	population.sort()
	return initialize_network(population[0].get_genes())

def evolve(mating_pool, elites, pop_size, cr, mr):
	new_population = []
	new_population += elites
	while len(new_population) < pop_size:
		p_a_idx = random.randrange(len(mating_pool))
		p_b_idx = random.randrange(len(mating_pool))
		if p_a_idx == p_b_idx:
			continue
		parent_a = mating_pool[p_a_idx]
		parent_b = mating_pool[p_b_idx]
		child_a, child_b = crossover(parent_a, parent_b, cr)
		child_a = mutation(child_a, mr)
		child_b = mutation(child_b, mr)
		new_population += [child_a, child_b]
	return new_population

def crossover(parent_a, parent_b, cr):
	if random.uniform(0.00, 1.00) >= cr:
		child_a = Chromosome(parent_a.get_genes(), parent_a.get_fit())
		child_b = Chromosome(parent_b.get_genes(), parent_b.get_fit())
		return child_a, child_b
	genes_a, genes_b = [], []
	pivot_a = random.randint(1, len(parent_a.get_genes())-1)
	pivot_b = random.randint(pivot_a, len(parent_a.get_genes())-1)
	for i in range(0, len(parent_a.get_genes())):
		if i < pivot_a:
			genes_a.append(parent_a[i])
			genes_b.append(parent_b[i])
		elif i < pivot_b:
			genes_a.append(parent_b[i])
			genes_b.append(parent_a[i])
		else:
			genes_a.append(parent_a[i])
			genes_b.append(parent_b[i])
	return Chromosome(genes_a), Chromosome(genes_b)

def mutation(child, mr):
	genes = [gene for gene in child.get_genes()]
	avg = sum(genes) / len(genes)
	for i in range(len(genes)):
		if random.uniform(0.00, 1.00) <= mr:
			genes[i] = random.gauss(mu=avg, sigma=0.9)
	if genes != child.get_genes():
		child.set_genes(genes)
	return child

def initialize_population(size, dim):
	population = []
	for _ in range(size):
		genes = [random.uniform(-0.50, 0.50) for _ in range(dim)]
		chromosome = Chromosome(genes)
		population.append(chromosome)
	return population

def elite_selection(population, percent):
	elites = []
	for i in range(ceil(len(population)*percent)):
		elites.append(population[i])
	return elites
	
def tournament_selection(population, percent):
	tournament = []
	for i in range(ceil(len(population)*percent)):
		random_idx = random.randint(0, len(population)-1)
		tournament.append(population.pop(random_idx))
	tournament.sort()
	return tournament[0]

def initialize_network(c):
	n, h, o = FEATURES, HIDDEN_SIZE, CLASSES
	chr = iter(c)
	neural_network = []
	neural_network.append([[next(chr) for i in range(n+1)] for j in range(h)])
	neural_network.append([[next(chr) for i in range(h+1)] for j in range(o)])
	return neural_network

def feed_forward(network, example):
	layer_input, layer_output = example, []
	for layer in network:
		for neuron in layer:
			summ = summing_function(neuron, layer_input)
			layer_output.append(activation_function(summ))
		layer_input, layer_output = layer_output, []
	return layer_input

def summing_function(weights, inputs):
	"""Sums the synapse weights with inputs and bias.
	
	Parameters:
		weights : synaptic weights.
		inputs : a vector of inputs.
	
	Returns:
		The aggregate of inputs times weights, plus bias.
	"""
	bias = weights[-1]
	summ = 0.00
	for i in range(len(weights)-1):
		summ += (weights[i] * float(inputs[i]))
	return summ + bias

def activation_function(z):
	return z if z >= 0 else 0

def performance_measure(chromosome, data):
	network = initialize_network(chromosome)
	correct, total = 0, 0
	for example in data:
		if check_output(network, example) == float(example[-1]):
			correct += 1
		total += 1
	return 100*(correct / total)

def check_output(network, example):
	output = feed_forward(network, example)
	return output.index(max(output))

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
	MSE, TRP, TEP = [], [], []
	TRAIN, TEST = load_data('../data/wheat.csv')
	FEATURES = len(TRAIN[0][:-1])
	CLASSES = len(list(set([c[-1] for c in (TRAIN+TEST)])))
	HIDDEN_SIZE = 8
	CHROMOSOME_SIZE = (HIDDEN_SIZE * (FEATURES+1)) + \
		(CLASSES * (HIDDEN_SIZE+1))
	POP_SIZE = 100
	CROSS_RATE, MUTAT_RATE = 0.90, 0.05
	ELITE_PROPORTION, TOURN_PROPORTION = 0.05, 0.03
	EPOCHS = 200
	NETWORK = genetic_network(ELITE_PROPORTION, TOURN_PROPORTION, \
		CHROMOSOME_SIZE, EPOCHS, POP_SIZE, CROSS_RATE, MUTAT_RATE)
	plot_data()

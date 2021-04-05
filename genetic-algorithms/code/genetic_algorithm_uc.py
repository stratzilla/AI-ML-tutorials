#!/usr/bin/env python3

from math import ceil
import random

class Chromosome:
	def __init__(self, genes, fit=None):
		self.genes = genes
		self.fit = fitness(self.genes) \
			if fit is None else fit

	def set_genes(self, genes):
		self.genes = genes
		self.fit = fitness(self.genes)
	
	def get_genes(self):
		return self.genes
	
	def get_fit(self):
		return self.fit

	def __lt__(self, other):
		return self.fit < other.fit
	
	def __getitem__(self, key):
		return self.genes[key]

def genetic_algorithm(el_p, to_p, dim, epochs, pop_size, s_min, s_max, cr, mr):
	population = initialize_population(pop_size, dim, s_min, s_max)
	for e in range(1, epochs+1):
		population.sort()
		mating_pool = []
		elites = elite_selection(population, el_p)
		del population[:len(elites)]
		t_winner = tournament_selection(population, to_p)
		mating_pool.extend(elites)
		mating_pool.append(t_winner)
		population = evolve(mating_pool, elites, pop_size, s_min, s_max, cr, mr)
		mating_pool.clear()
	population.sort()
	best_fitness = fitness(population[0])
	print(f'Best minimum found was {best_fitness:.2f}.')

def evolve(mating_pool, elites, pop_size, s_min, s_max, cr, mr):
	new_population = []
	new_population.extend(elites)
	while len(new_population) < pop_size:
		p_a_idx = random.randrange(len(mating_pool))
		p_b_idx = random.randrange(len(mating_pool))
		if p_a_idx == p_b_idx:
			continue
		parent_a = mating_pool[p_a_idx]
		parent_b = mating_pool[p_b_idx]
		child_a, child_b = crossover(parent_a, parent_b, cr)
		child_a = mutation(child_a, s_min, s_max, mr)
		child_b = mutation(child_b, s_min, s_max, mr)
		new_population.append(child_a)
		new_population.append(child_b)
	return new_population

def crossover(parent_a, parent_b, cr):
	if random.uniform(0.00, 1.00) >= cr:
		child_a = Chromosome(parent_a.get_genes(), parent_a.get_fit())
		child_b = Chromosome(parent_b.get_genes(), parent_b.get_fit())
		return child_a, child_b
	genes_a, genes_b = [], []
	parent_a_genes = parent_a.get_genes()
	parent_b_genes = parent_b.get_genes()
	pivot = random.randint(1, len(parent_a_genes)-1)
	for i in range(0, len(parent_a_genes)):
		if i < pivot:
			genes_a.append(parent_a_genes[i])
			genes_b.append(parent_b_genes[i])
		else:
			genes_a.append(parent_b_genes[i])
			genes_b.append(parent_a_genes[i])
	return Chromosome(genes_a), Chromosome(genes_b)

def mutation(child, s_min, s_max, mr):
	genes = [gene for gene in child.get_genes()]
	for i in range(len(genes)):
		if random.uniform(0.00, 1.00) <= mr:
			genes[i] += random.uniform(s_min*0.50, s_max*0.50)
			genes[i] = max(genes[i], s_min)
			genes[i] = min(genes[i], s_max)
	if genes != child.get_genes():
		child.set_genes(genes)
	return child

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

def fitness(genes):
	summ = 0
	for gene in genes:
		summ += gene**4 - (16 * gene**2) + (5 * gene)
	return (1 / 2) * summ

def initialize_population(size, dim, s_min, s_max):
	population = []
	for _ in range(size):
		genes = [random.uniform(s_min, s_max) for _ in range(dim)]
		chromosome = Chromosome(genes)
		population.append(chromosome)
	return population

if __name__ == '__main__':
	SEARCH_MIN, SEARCH_MAX = -5.00, 5.00
	DIMENSIONS = 2
	POP_SIZE = 200
	ELITE_PROPORTION = 0.01
	TOURN_PROPORTION = 0.02
	EPOCHS = 10
	CROSS_RATE, MUTAT_RATE = 0.95, 0.05
	genetic_algorithm(ELITE_PROPORTION, TOURN_PROPORTION, DIMENSIONS, EPOCHS, \
		POP_SIZE, SEARCH_MIN, SEARCH_MAX, CROSS_RATE, MUTAT_RATE)
#!/usr/bin/env python3

from math import ceil
import random

class Chromosome:
	"""Chromosome class.
	Containerizes genes for chromosome.
	
	Attributes:
		pos : the position in n-space.
		fit : the fitness of the chromosome.
	"""
	
	def __init__(self, genes, fit=None):
		"""Chromosome constructor without fitness."""
		# initialize position from parameter
		self.genes = genes
		# if no argument passed as fitness
		# take fitness from genes argument
		# else init as fit argument
		self.fit = fitness(self.genes) \
			if fit is None else fit
	
	def set_genes(self, genes):
		"""Genes mutator method."""
		self.genes = genes
		# when setting genes subsequent times
		# update the fitness
		self.fit = fitness(self.genes)
	
	def get_genes(self):
		"""Genes accessor method."""
		return self.genes
	
	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

	def __lt__(self, other):
		"""Less-than operator overload."""
		return self.fit < other.fit
	
	def __getitem__(self, key):
		"""List index operator overload."""
		return self.genes[key]

def genetic_algorithm(el_p, to_p, dim, epochs, pop_size, s_min, s_max, cr, mr):
	"""Genetic algorithm driver.
	Initializes the population and evolves it over time.
	
	Parameters:
		el_p : the proportion of elites for elite selection.
		to_p : the proportion of tournament entrants for tournament selection.
		dim : dimensionality of the problem.
		epochs : how many iterations of the evolution algorithm to perform.
		function : the fitness function to evolve from.
	"""
	# initially the population is random
	population = initialize_population(pop_size, dim, s_min, s_max)
	for e in range(1, epochs+1): # for each iteration
		population.sort() # sort by fitness
		mating_pool = [] # init empty mating pool
		# select elites based on best fitness in population
		elites = elite_selection(population, el_p)
		# delete elites from population
		del population[:len(elites)]
		# select tournament winner from random candidates
		t_winner = tournament_selection(population, to_p)
		# add the elites to the mating pool
		mating_pool.extend(elites)
		# add the tournament winner to the mating pool
		mating_pool.append(t_winner)
		# evole the population based on parents and genetic operators
		population = evolve(mating_pool, elites, pop_size, s_min, s_max, cr, mr)
		mating_pool.clear() # erase the mating pool for next generation
	# sort population by fitness again
	population.sort()
	# return the best fitness after the entire genetic algorithm
	best_fitness = fitness(population[0])
	print(f'Best minimum found was {best_fitness:.2f}.')

def evolve(mating_pool, elites, pop_size, s_min, s_max, cr, mr):
	"""Evolves population based on genetic operators.
	
	Parameters:
		mating_pool : where to select parents from.
		elites : previously found elites.
		pop_size : the population size.
		s_min : search space minimum bound.
		s_max : search space maximum bound.
		cr : crossover rate.
		mr : mutation rate.
	
	Returns:
		A new population of offspring from mating pool.
	"""
	new_population = [] # store new population as list
	new_population.extend(elites) # add elites verbatim
	while len(new_population) < pop_size: # while population isn't at max size
		# get both parents indices
		p_a_idx = random.randrange(len(mating_pool))
		p_b_idx = random.randrange(len(mating_pool))
		# we don't mind parents have identical genes but we don't
		# want the parents to use the same index. Parent A can be
		# equal to Parent B, but Parent A cannot be Parent B
		if p_a_idx == p_b_idx:
			continue
		# get the parents from indices
		parent_a = mating_pool[p_a_idx]
		parent_b = mating_pool[p_b_idx]
		# find children using crossover
		child_a, child_b = crossover(parent_a, parent_b, cr)
		# mutate each child
		child_a = mutation(child_a, s_min, s_max, mr)
		child_b = mutation(child_b, s_min, s_max, mr)
		# add children to population
		new_population.append(child_a)
		new_population.append(child_b)
	return new_population

def crossover(parent_a, parent_b, cr):
	"""One-point crossover operator.
	
	Parameters:
		parent_a : the first parent.
		parent_b : the second parent.
		cr : the crossover chance.
	
	Returns:
		Two child chromosomes as a product of both parents.
	"""
	# only perform crossover based on the crossover rate
	if random.uniform(0.00, 1.00) >= cr:
		child_a = Chromosome(parent_a.get_genes(), parent_a.get_fit())
		child_b = Chromosome(parent_b.get_genes(), parent_b.get_fit())
		return child_a, child_b
	genes_a, genes_b = [], []
	parent_a_genes = parent_a.get_genes()
	parent_b_genes = parent_b.get_genes()
	# find a pivot point at random
	pivot = random.randint(1, len(parent_a_genes)-1)
	for i in range(0, len(parent_a_genes)):
		# before pivot, use genes from one parent
		if i < pivot:
			genes_a.append(parent_a_genes[i])
			genes_b.append(parent_b_genes[i])
		# after pivot, use genes from second parent
		else:
			genes_a.append(parent_b_genes[i])
			genes_b.append(parent_a_genes[i])
	return Chromosome(genes_a), Chromosome(genes_b)

def mutation(child, s_min, s_max, mr):
	"""Mutation operator.
	
	Parameters:
		child : the chromosome to mutate.
		s_min : the lower bound for mutation.
		s_max : the upper bound for mutation.
		mr : mutation chance.
	
	Returns:
		A mutated child.
	"""
	# the new genes to make
	genes = [gene for gene in child.get_genes()]
	for i in range(len(genes)):
		# only perform mutation based on the mutation rate
		if random.uniform(0.00, 1.00) <= mr:
			# update the gene with random mutation
			# the mutation is within half the radius of the search space
			genes[i] += random.uniform(s_min*0.50, s_max*0.50)
			# the position is now randomized but still relatively close
			# if it's outside the search space, clamp it to within
			genes[i] = max(genes[i], s_min)
			genes[i] = min(genes[i], s_max)
	# we don't need to update the fitness if the gene
	# hasn't changed, so only update genes if they've changed
	if genes != child.get_genes():
		child.set_genes(genes)
	return child

def elite_selection(population, percent):
	"""Elite selection function.
	Stores elites to bring into the next generation and mating pool.
	
	Parameters:
		population : the population to take elites from.
		percent : the proportion of the population to consider elites.
	
	Returns:
		A list of elite solutions.
	"""
	elites = []
	# grab percent% best individuals
	for i in range(ceil(len(population)*percent)):
		elites.append(population[i]) # and append to elites
	return elites

def tournament_selection(population, percent):
	"""Tournament selection function.
	Creates a tournament of random individuals and returns the best.
	
	Parameters:
		population : the population to take tournament from.
		percent : the proportion of the population who enters the tournament.
	
	Returns:
		Best fit individual from tournament.
	"""
	tournament = []
	# grab percent% random individuals
	for i in range(ceil(len(population)*percent)):
		random_idx = random.randint(0, len(population)-1)
		tournament.append(population.pop(random_idx)) # append to tournament
	tournament.sort() # sort by fitness
	return tournament[0] # return best fit from tournament

def fitness(genes):
	"""GA fitness function.
	Uses Styblinski-Tang Function in d dimensions.
	
	Parameters:
		genes : the genes to get fitness of.
	
	Returns:
		The fitness of that gene.
	"""
	summ = 0
	for gene in genes:
		summ += gene**4 - (16 * gene**2) + (5 * gene)
	return (1 / 2) * summ

def initialize_population(size, dim, s_min, s_max):
	"""Initializes a random population.
	
	Parameters:
		size : the size of the population.
		dim : the dimensionality of the problem
		s_min : the minimum in a dimension for the search space.
		s_max : a maximum in a dimension for the search space.
	
	Returns:
		Random population of that many chromosomes within search space.
	"""
	population = [] # population stored as a list
	for _ in range(size): # for the size of the population
		# make random genes
		genes = [random.uniform(s_min, s_max) for _ in range(dim)]
		chromosome = Chromosome(genes) # create the chromosome
		population.append(chromosome) # add to population
	return population

if __name__ == '__main__':
	# bounds for the search space
	SEARCH_MIN, SEARCH_MAX = -5.00, 5.00
	DIMENSIONS = 2 # dimensionality of the problem
	POP_SIZE = 200 # how big is the population
	ELITE_PROPORTION = 0.01 # proportion of elites
	TOURN_PROPORTION = 0.02 # proportion of tournament
	EPOCHS = 10 # how many generations to run for
	# generatic operator chances
	CROSS_RATE, MUTAT_RATE = 0.95, 0.05
	genetic_algorithm(ELITE_PROPORTION, TOURN_PROPORTION, DIMENSIONS, EPOCHS, \
		POP_SIZE, SEARCH_MIN, SEARCH_MAX, CROSS_RATE, MUTAT_RATE)
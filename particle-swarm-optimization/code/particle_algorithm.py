#!/usr/bin/env python3

from math import sin, sqrt
import random

class Particle:
	"""Particle class.
	Containzerizes a position, velocity.
	
	Attributes:
		pos : the position in n-space.
		best_pos : the best position this particle has had.
		fit : the fitness of the particle.
		best_fit : the best fitness this particle has had.
		vel : the velocity in n-space.
	"""
	
	def __init__(self, pos, vel):
		"""Particle constructor."""
		# initialize position and velocity as params
		self.pos, self.vel = pos, vel
		# find fitness at instantiation
		self.fit = schwefel(self.pos)
		# best so far is just initial
		self.best_pos, self.best_fit = self.pos, self.fit
	
	def set_pos(self, pos):
		"""Position mutator method."""
		self.pos = pos
		# get fitness of new position if the new
		# position is still within the search
		# space. Otherwise don't update fitness
		if not any(p < SEARCH_MIN for p in pos)\
		and not any(p > SEARCH_MAX for p in pos):
			# get the fitness
			fitness = schwefel(self.pos)
			# if better
			if fitness < self.best_fit:
				self.fit = fitness
				# update best fitness
				self.best_fit = self.fit
				# update best position
				self.best_pos = self.pos
	
	def set_vel(self, vel):
		"""Velocity mutator method."""
		self.vel = vel
	
	def get_pos(self):
		"""Position accessor method."""
		return self.pos
	
	def get_vel(self):
		"""Velocity accessor method."""
		return self.vel
	
	def get_best_pos(self):
		"""Best position accessor method."""
		return self.best_pos

	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

def pso(dim, epochs):
	"""Particle Swarm Optimization driver.
	Initializes the swarm and improves it over time.
	
	Parameters:
		dim : the dimensionality of the problem.
		epochs : how many iterations.
	"""
	# initialize the swarm based on problem scope and swarm size
	swarm = initialize_swarm(SWARM_SIZE, dim, SEARCH_MIN, SEARCH_MAX)
	for e in range(1, epochs+1): # for each iteration
		move_particles(swarm) # move particles to more fit neighborhood
	# return the swarm best fitness
	best_fitness = get_swarm_best(swarm)[0]
	print(f'Best minimum found was {best_fitness:.2f}.')

def move_particles(swarm):
	"""Particle movement function.
	
	Parameters:
		swarm : the swarm to move.
	"""
	# get swarm bests
	swarm_best = get_swarm_best(swarm)
	for particle in swarm: # for each particle
		# new position and velocity is initially zero
		new_pos = [0 for _ in range(DIMENSIONS)]
		new_vel = [0 for _ in range(DIMENSIONS)]
		for d in range(DIMENSIONS): # for each axis
			# the social and cognitive coefficients 
			# take a stochastic multiplicant
			R_1 = random.uniform(0.00, 1.00)
			R_2 = random.uniform(0.00, 1.00)
			# this is split for readability but the update is based
			# on an addition of a weight, cognitive, and social term
			weight = W * particle.get_vel()[d]
			cognitive = C_1 * R_1
			cognitive *= (particle.get_best_pos()[d] - particle.get_pos()[d])
			social = C_2 * R_2
			social *= (swarm_best[1][d] - particle.get_pos()[d])
			# new velocity is simply weight + cognitive + social
			new_vel[d] = weight + cognitive + social
			# new position is just old position + velocity
			new_pos[d] = particle.get_pos()[d] + new_vel[d]
		# update particle with new position and velocity
		particle.set_pos(new_pos)
		particle.set_vel(new_vel)    

def initialize_swarm(size, dim, min, max):
	"""Swarm initialization function.
	
	Parameters:
		size : the size of our swarm.
		dim : the dimensionality of the problem.
		min : the minimum in a dimension for the search space.
		max : the maximum in a dimension for the search space.
	
	Returns:
		A random swarm of that many Particles within the bounds of the space.
	"""
	swarm = [] # swarm stored as list
	for _ in range(size): # for the size of the swarm
		# position is random in every dimension
		position = [random.uniform(min, max) for _ in range(dim)]
		# velocity is initially zero in every dimension
		velocity = [0 for _ in range(dim)]
		# init a particle
		particle = Particle(position, velocity)
		swarm.append(particle) # add to swarm
	return swarm

def get_swarm_best(swarm):
	"""Finds the swarm best fitness and position.
	
	Parameters:
		swarm : the swarm to search.
		
	Returns:
		The swarm best fitness and swarm best position.
	"""
	# initially assume the first is the best
	best_fit = swarm[0].get_fit()
	best_pos = swarm[0].get_pos()
	for particle in swarm: # for each particle
		# if better fitness found
		if particle.get_fit() < best_fit:
			# update best fitness and position
			best_fit = particle.get_fit()
			best_pos = particle.get_pos()
	return best_fit, best_pos

def schwefel(particle):
	"""PSO fitness function.
	Uses Schwefel function in d dimensions.
	
	Parameters:
		particle : a particle in d-space.
	
	Returns:
		The fitness of that particle.
	"""
	summ = 0
	for d in particle:
		summ += d * sin(sqrt(abs(d)))
	return (len(particle) * 418.9829) - summ

if __name__ == '__main__':
	SEARCH_MIN = -500 # minimum in search space
	SEARCH_MAX = 500 # maximum in search space
	DIMENSIONS = 2 # dimensionality of the problem
	SWARM_SIZE = 200 # how large our swarm is
	EPOCHS = 100 # how many iterations
	W = 0.729844 # intertial weight
	C_1 = 1.496180 # cognitive coefficient
	C_2 = 1.496180 # social coefficient
	pso(DIMENSIONS, EPOCHS)
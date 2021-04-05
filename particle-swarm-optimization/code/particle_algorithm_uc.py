#!/usr/bin/env python3

from math import sin, sqrt
import random

class Particle:
	def __init__(self, pos, vel):
		self.pos, self.vel = pos, vel
		self.fit = schwefel(self.pos)
		self.best_pos, self.best_fit = self.pos, self.fit
	
	def set_pos(self, pos):
		self.pos = pos
		if not any(p < SEARCH_MIN for p in pos)\
		and not any(p > SEARCH_MAX for p in pos):
			fitness = schwefel(self.pos)
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

def pso(dim, epochs):
	swarm = initialize_swarm(SWARM_SIZE, dim, SEARCH_MIN, SEARCH_MAX)
	for e in range(1, epochs+1):
		move_particles(swarm)
	best_fitness = get_swarm_best(swarm)[0]
	print(f'Best minimum found was {best_fitness:.2f}.')

def move_particles(swarm):
	swarm_best = get_swarm_best(swarm)
	for particle in swarm:
		new_pos = [0 for _ in range(DIMENSIONS)]
		new_vel = [0 for _ in range(DIMENSIONS)]
		for d in range(DIMENSIONS):
			R_1 = random.uniform(0.00, 1.00)
			R_2 = random.uniform(0.00, 1.00)
			weight = W * particle.get_vel()[d]
			cognitive = C_1 * R_1
			cognitive *= (particle.get_best_pos()[d] - particle.get_pos()[d])
			social = C_2 * R_2
			social *= (swarm_best[1][d] - particle.get_pos()[d])
			new_vel[d] = weight + cognitive + social
			new_pos[d] = particle.get_pos()[d] + new_vel[d]
		particle.set_pos(new_pos)
		particle.set_vel(new_vel)    

def initialize_swarm(size, dim, min, max):
	swarm = []
	for _ in range(size):
		position = [random.uniform(min, max) for _ in range(dim)]
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

def schwefel(particle):
	summ = 0
	for d in particle:
		summ += d * sin(sqrt(abs(d)))
	return (len(particle) * 418.9829) - summ

if __name__ == '__main__':
	SEARCH_MIN = -500
	SEARCH_MAX = 500
	DIMENSIONS = 2
	SWARM_SIZE = 200
	EPOCHS = 100
	W = 0.729844
	C_1 = 1.496180
	C_2 = 1.496180
	pso(DIMENSIONS, EPOCHS)
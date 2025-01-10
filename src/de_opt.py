import numpy as np
import matplotlib.pyplot as plt


class DEOptimizer:
    def __init__(self, objective_func, bounds,
                 colony_size=100, max_iterations=200):
        self.objective_func = objective_func
        self.bounds = bounds
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.dimension = len(bounds)

        self.F = 0.8  # mutation factor
        self.CR = 0.7  # crossover rate

        self.population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.colony_size, self.dimension)
        )

        self.fitness = np.array([self.objective_func(ind)
                                 for ind in self.population])

        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        self.best_solutions = []
        self.mean_solutions = []

    def optimize(self):
        for iteration in range(self.max_iterations):
            for i in range(self.colony_size):
                candidates = [idx for idx in range(
                    self.colony_size) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                mutant = self.population[a] + self.F * (
                    self.population[b] - self.population[c])

                mutant = np.clip(mutant,
                                 [b[0] for b in self.bounds],
                                 [b[1] for b in self.bounds])

                cross_points = np.random.rand(self.dimension) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = self.objective_func(trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial.copy()
                        self.best_fitness = trial_fitness

            current_best = np.min(self.fitness)
            current_mean = np.mean(self.fitness)
            self.best_solutions.append(current_best)
            self.mean_solutions.append(current_mean)

        return self.best_solution

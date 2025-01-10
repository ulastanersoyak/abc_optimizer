import numpy as np
import matplotlib.pyplot as plt


class PSOOptimizer:
    def __init__(self, objective_func, bounds,
                 colony_size=100, max_iterations=200):
        self.objective_func = objective_func
        self.bounds = bounds
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.dimension = len(bounds)

        self.w = 0.729  # inertia weight
        self.c1 = 1.49445  # cognitive parameter
        self.c2 = 1.49445  # social parameter

        self.positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.colony_size, self.dimension)
        )

        self.velocities = np.random.uniform(
            low=[-abs(b[1]-b[0])/4 for b in bounds],
            high=[abs(b[1]-b[0])/4 for b in bounds],
            size=(self.colony_size, self.dimension)
        )

        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([self.objective_func(p)
                                   for p in self.positions])

        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]

        self.best_solutions = []
        self.mean_solutions = []

    def optimize(self):
        for iteration in range(self.max_iterations):
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.pbest_pos - self.positions) +
                               self.c2 * r2 * (self.gbest_pos - self.positions))

            self.positions += self.velocities

            self.positions = np.clip(
                self.positions,
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds]
            )

            current_values = np.array([self.objective_func(p)
                                       for p in self.positions])

            improved = current_values < self.pbest_val
            self.pbest_pos[improved] = self.positions[improved]
            self.pbest_val[improved] = current_values[improved]

            min_idx = np.argmin(self.pbest_val)
            if self.pbest_val[min_idx] < self.gbest_val:
                self.gbest_pos = self.pbest_pos[min_idx].copy()
                self.gbest_val = self.pbest_val[min_idx]

            current_best = np.min(current_values)
            current_mean = np.mean(current_values)
            self.best_solutions.append(current_best)
            self.mean_solutions.append(current_mean)

        return self.gbest_pos

import numpy as np
from visualization import OptimizationVisualizer


class ABCOptimizer:
    def __init__(self, objective_func, bounds,
                 colony_size=100, max_iterations=200):
        self.objective_func = objective_func
        self.bounds = bounds
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.dimension = len(bounds)
        self.employed_bees = self.colony_size // 2
        self.onlooker_bees = self.colony_size // 2
        self.limit = self.colony_size * self.dimension
        self.trial = np.zeros(self.employed_bees)

        # Initialize food sources
        self.food_sources = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.employed_bees, self.dimension)
        )
        self.fitness = np.array([self.calculate_fitness(
            self.objective_func(x)) for x in self.food_sources])

    def calculate_fitness(self, fx):
        if fx >= 0:
            return 1 / (1 + fx)
        else:
            return 1 + abs(fx)

    def employed_bee_phase(self):
        for i in range(self.employed_bees):
            k = np.random.choice(
                [x for x in range(self.employed_bees) if x != i])
            dims_to_modify = np.random.randint(1, self.dimension + 1)
            dims = np.random.choice(
                self.dimension, dims_to_modify, replace=False)

            new_position = self.food_sources[i].copy()
            for j in dims:
                phi = np.random.uniform(-1, 1)
                new_position[j] = self.food_sources[i][j] + phi * \
                    (self.food_sources[i][j] - self.food_sources[k][j])

            new_position = np.clip(new_position,
                                   [b[0] for b in self.bounds],
                                   [b[1] for b in self.bounds])

            new_fitness = self.calculate_fitness(
                self.objective_func(new_position))

            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_position
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def onlooker_bee_phase(self):
        probabilities = self.fitness / np.sum(self.fitness)

        for _ in range(self.onlooker_bees):
            i = np.random.choice(range(self.employed_bees), p=probabilities)
            k = np.random.choice(
                [x for x in range(self.employed_bees) if x != i])

            dims_to_modify = np.random.randint(1, self.dimension + 1)
            dims = np.random.choice(
                self.dimension, dims_to_modify, replace=False)

            new_position = self.food_sources[i].copy()
            for j in dims:
                phi = np.random.uniform(-1, 1)
                new_position[j] = self.food_sources[i][j] + phi * \
                    (self.food_sources[i][j] - self.food_sources[k][j])

            new_position = np.clip(new_position,
                                   [b[0] for b in self.bounds],
                                   [b[1] for b in self.bounds])

            new_fitness = self.calculate_fitness(
                self.objective_func(new_position))

            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_position
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def scout_bee_phase(self):
        for i in range(self.employed_bees):
            if self.trial[i] >= self.limit:
                best_idx = np.argmax(self.fitness)
                std = (self.bounds[0][1] - self.bounds[0][0]) / 4

                new_position = np.random.normal(
                    loc=self.food_sources[best_idx],
                    scale=std,
                    size=self.dimension
                )

                new_position = np.clip(new_position,
                                       [b[0] for b in self.bounds],
                                       [b[1] for b in self.bounds])

                self.food_sources[i] = new_position
                self.fitness[i] = self.calculate_fitness(
                    self.objective_func(new_position))
                self.trial[i] = 0

    def optimize(self):
        visualizer = OptimizationVisualizer(self.bounds, self.objective_func)
        best_solution_ever = None
        best_fitness_ever = float('-inf')

        for iteration in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            current_objectives = [self.objective_func(
                x) for x in self.food_sources]
            best_solution = np.min(current_objectives)
            mean_solution = np.mean(current_objectives)

            current_best_idx = np.argmin(current_objectives)
            if self.fitness[current_best_idx] > best_fitness_ever:
                best_fitness_ever = self.fitness[current_best_idx]
                best_solution_ever = self.food_sources[current_best_idx].copy()

            visualizer.update(iteration, self.food_sources,
                              best_solution, mean_solution)

        visualizer.show()
        return best_solution_ever

import numpy as np
import matplotlib.pyplot as plt


class abc_optimizer:
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

        self.best_solutions = []
        self.mean_solutions = []

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
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.objective_func([i, j]) for i in x] for j in y])

        contour = ax1.contour(X, Y, Z, levels=50)
        ax1.clabel(contour, inline=True, fontsize=8)
        scatter = ax1.scatter([], [], c='red', marker='x')
        ax1.set_title('Food Source Positions')

        line1, = ax2.plot([], [], 'b-', label='Best Fitness')
        line2, = ax2.plot([], [], 'r-', label='Mean Fitness')
        ax2.set_xlim(0, self.max_iterations)
        ax2.set_ylim(0, 1)
        ax2.set_title('Convergence Plot')
        ax2.legend()

        x_data = []
        best_data = []
        mean_data = []

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

            x_data.append(iteration)
            best_data.append(self.calculate_fitness(best_solution))
            mean_data.append(self.calculate_fitness(mean_solution))

            scatter.set_offsets(self.food_sources)
            line1.set_data(x_data, best_data)
            line2.set_data(x_data, mean_data)

            plt.draw()
            plt.pause(0.05)

        plt.ioff()
        plt.show()

        return best_solution_ever


def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


bounds = [(-5.12, 5.12), (-5.12, 5.12)]

optimizer = abc_optimizer(
    objective_func=rastrigin_function,
    bounds=bounds,
    colony_size=100,
    max_iterations=100
)

best_solution = optimizer.optimize()
print(f"Best solution found: {best_solution}")
print(f"Best fitness: {rastrigin_function(best_solution)}")

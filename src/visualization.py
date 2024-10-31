import matplotlib.pyplot as plt
import numpy as np


class OptimizationVisualizer:
    def __init__(self, bounds, objective_func):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Setup contour plot
        x = np.linspace(bounds[0][0], bounds[0][1], 200)
        y = np.linspace(bounds[1][0], bounds[1][1], 200)
        X, Y = np.meshgrid(x, y)

        # Create evaluation points as numpy arrays
        Z = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))

        self.contour = self.ax1.contour(X, Y, Z, levels=50)
        self.ax1.clabel(self.contour, inline=True, fontsize=8)
        self.scatter = self.ax1.scatter([], [], c='red', marker='x')
        self.ax1.set_title('Food Source Positions')

        # Setup convergence plot
        self.line1, = self.ax2.plot([], [], 'b-', label='Best Fitness')
        self.line2, = self.ax2.plot([], [], 'r-', label='Mean Fitness')
        self.ax2.set_xlim(0, 200)  # Assuming max_iterations = 200
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Convergence Plot')
        self.ax2.legend()

        self.x_data = []
        self.best_data = []
        self.mean_data = []

    def update(self, iteration, food_sources, best_solution, mean_solution):
        self.x_data.append(iteration)

        self.best_data.append(1 / (1 + best_solution)
                              if best_solution >= 0 else 1 +
                              abs(best_solution))

        self.mean_data.append(1 / (1 + mean_solution)
                              if mean_solution >= 0 else 1 +
                              abs(mean_solution))

        self.scatter.set_offsets(food_sources)
        self.line1.set_data(self.x_data, self.best_data)
        self.line2.set_data(self.x_data, self.mean_data)

        plt.draw()
        plt.pause(0.05)

    def show(self):
        plt.ioff()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
import time

from abc_opt import ABCOptimizer
from pso_opt import PSOOptimizer
from de_opt import DEOptimizer
from objectives import rastrigin_function, sphere_function, rosenbrock_function


def run_experiment(optimizer_class, objective_func, bounds, runs=30):
    best_fitnesses = []
    execution_times = []
    all_histories = []

    for run in range(runs):
        start_time = time.time()
        optimizer = optimizer_class(
            objective_func=objective_func,
            bounds=bounds,
            colony_size=100,
            max_iterations=200
        )

        best_solution = optimizer.optimize()
        execution_time = time.time() - start_time

        best_fitness = objective_func(best_solution)
        best_fitnesses.append(best_fitness)
        execution_times.append(execution_time)
        all_histories.append(optimizer.best_solutions)

    return {
        'best_fitnesses': best_fitnesses,
        'execution_times': execution_times,
        'convergence_histories': all_histories
    }


def plot_convergence(histories, title):
    plt.figure(figsize=(10, 6))
    histories = np.array(histories)
    mean_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)

    plt.plot(mean_history, label='Mean Best Fitness')
    plt.fill_between(
        range(len(mean_history)),
        mean_history - std_history,
        mean_history + std_history,
        alpha=0.2
    )

    plt.title(f'Convergence Plot - {title}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()


def main():
    test_functions = {
        'Rastrigin': (rastrigin_function, [(-5.12, 5.12), (-5.12, 5.12)]),
        'Sphere': (sphere_function, [(-100, 100), (-100, 100)]),
        'Rosenbrock': (rosenbrock_function, [(-2.048, 2.048), (-2.048, 2.048)])
    }

    optimizers = {
        'ABC': ABCOptimizer,
        'PSO': PSOOptimizer,
        'DE': DEOptimizer
    }

    results = {}

    for func_name, (func, bounds) in test_functions.items():
        print(f"\nTesting {func_name} function:")
        results[func_name] = {}

        for opt_name, optimizer in optimizers.items():
            print(f"\nRunning {opt_name}...")
            results[func_name][opt_name] = run_experiment(
                optimizer, func, bounds)

            best_fitnesses = results[func_name][opt_name]['best_fitnesses']
            times = results[func_name][opt_name]['execution_times']

            print(f"Best fitness: {np.min(best_fitnesses):.2e}")
            print(f"Mean fitness: {np.mean(best_fitnesses):.2e}")
            print(f"Std fitness: {np.std(best_fitnesses):.2e}")
            print(f"Mean time: {np.mean(times):.3f} seconds")

            plt.figure()
            plot_convergence(
                results[func_name][opt_name]['convergence_histories'],
                f"{func_name} - {opt_name}"
            )
            plt.savefig(f"{func_name}_{opt_name}_convergence.png")
            plt.close()

        plt.figure(figsize=(12, 6))
        for opt_name in optimizers.keys():
            histories = np.array(
                results[func_name][opt_name]['convergence_histories'])
            mean_history = np.mean(histories, axis=0)
            plt.plot(mean_history, label=opt_name)

        plt.title(f'Convergence Comparison - {func_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{func_name}_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()

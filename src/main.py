from optimizer import ABCOptimizer
from objectives import rastrigin_function, sphere_function, rosenbrock_function
import numpy as np

# Dictionary of standard test functions and their bounds
TEST_FUNCTIONS = {
    'rastrigin': {
        'function': rastrigin_function,
        'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
        'global_minimum': 0.0,
        'global_minimum_position': [0.0, 0.0]
    },
    'sphere': {
        'function': sphere_function,
        # Can also use [(-100, 100), (-100, 100)]
        'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
        'global_minimum': 0.0,
        'global_minimum_position': [0.0, 0.0]
    },
    'rosenbrock': {
        'function': rosenbrock_function,
        'bounds': [(-2.048, 2.048), (-2.048, 2.048)],  # Classic bounds
        # Alternative bounds: [(-5, 10), (-5, 10)]
        'global_minimum': 0.0,
        'global_minimum_position': [1.0, 1.0]
    }
}


def main():
    function_name = 'rosenbrock'

    func_props = TEST_FUNCTIONS[function_name]

    optimizer = ABCOptimizer(
        objective_func=func_props['function'],
        bounds=func_props['bounds'],
        colony_size=100,
        max_iterations=200
    )

    best_solution = optimizer.optimize()
    best_fitness = func_props['function'](best_solution)

    print(f"\nOptimizing {function_name} function:")
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Known global minimum: {func_props['global_minimum']}")
    print(f"Known optimal position: {func_props['global_minimum_position']}")
    print(f"Distance to optimal: {np.linalg.norm(
        best_solution - np.array(func_props['global_minimum_position']))}")


if __name__ == "__main__":
    main()

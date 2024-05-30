import random

import numpy as np

from elemelek.logging import SelfLogging


class OptimalSubsetGeneticAlgorithm(SelfLogging):
    def __init__(
        self,
        distance_matrix: np.ndarray,
        target_distance: float,
        population_size: int,
        sample_size: int,
        generations: int,
        mutation_rate: float,
    ):
        self.distance_matrix = distance_matrix
        self.target_distance = target_distance
        self.population_size = population_size
        self.sample_size = sample_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _init_solution(self) -> np.ndarray:
        solution = np.zeros(len(self.distance_matrix), dtype=np.uint8)
        indices = np.random.choice(
            np.arange(len(solution)), self.sample_size, replace=False
        )
        solution[indices] = 1

        assert len(solution.nonzero()[0]) == self.sample_size
        return solution

    def _eval(self, solution: np.ndarray) -> float:
        submatrix = self.distance_matrix[
            np.ix_(solution.nonzero()[0], solution.nonzero()[0])
        ]

        return abs(np.median(submatrix) - self.target_distance)

    def _cross(self, solution_a: np.ndarray, solution_b: np.ndarray) -> np.ndarray:
        non_zero_indices = np.nonzero(solution_a + solution_b)[0]
        new_solution = np.zeros_like(solution_a, dtype=np.uint8)
        chosen = np.random.choice(
            non_zero_indices, size=self.sample_size, replace=False
        )

        new_solution[chosen] = 1
        assert len(new_solution.nonzero()[0]) == self.sample_size
        return new_solution

    def _mutate(self, solution: np.ndarray) -> np.ndarray:
        mutation_indices = np.random.choice(
            np.arange(len(solution)), size=self.sample_size, replace=False
        )
        for index in mutation_indices:
            solution[index] = 1 - solution[index]

        ones = np.where(solution == 1)[0]
        zeros = np.where(solution == 0)[0]
        if len(ones) > self.sample_size:
            drop_indices = np.random.choice(
                ones, size=len(ones) - self.sample_size, replace=False
            )
            solution[drop_indices] = 0
        elif len(ones) < self.sample_size:
            add_indices = np.random.choice(
                zeros, size=self.sample_size - len(ones), replace=False
            )
            solution[add_indices] = 1

        assert len(solution.nonzero()[0]) == self.sample_size
        return solution

    def optimize(self):
        population = [self._init_solution() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = np.inf

        for generation in range(self.generations):
            fitness = np.array([self._eval(sol) for sol in population])
            min_index = np.argmin(fitness)
            if fitness[min_index] < best_fitness:
                best_fitness = fitness[min_index]
                best_solution = population[min_index]

            new_population = []
            while len(new_population) < self.population_size:
                parents = random.sample(population, 2)
                offspring = self._cross(parents[0], parents[1])
                if random.random() < self.mutation_rate:
                    offspring = self._mutate(offspring)
                new_population.append(offspring)
            population = new_population
        assert len(best_solution.nonzero()[0]) == self.sample_size
        return best_solution.nonzero()[0]

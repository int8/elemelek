import random

import numpy as np

from elemelek.logging import SelfLogging


class OptimalSubsetGeneticAlgorithm(SelfLogging):
    def __init__(
        self,
        similarity_matrix: np.ndarray,
        target_similarity_median: float,
        population_size: int,
        sample_size: int,
        generations: int,
        mutation_rate: float,
    ):
        self.similarity_matrix = similarity_matrix
        self.target_similarity_median = target_similarity_median
        self.population_size = population_size
        self.sample_size = sample_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _init_solution(self):
        solution = np.zeros(len(self.similarity_matrix), dtype=np.uint8)
        indices = random.sample(range(len(self.similarity_matrix)), self.sample_size)
        solution[indices] = 1
        return solution

    def _eval(self, solution):
        submatrix = self.similarity_matrix[
            np.ix_(solution.nonzero()[0], solution.nonzero()[0])
        ]

        return abs(np.median(submatrix) - self.target_similarity_median)

    def _cross(self, solution_a, solution_b):
        non_zero_indices = np.nonzero(solution_a + solution_b)[0]
        if len(non_zero_indices) < self.sample_size:
            non_zero_indices = np.pad(
                non_zero_indices, (0, self.sample_size - len(non_zero_indices)), "wrap"
            )
        new_solution = np.zeros(len(self.similarity_matrix), dtype=np.uint8)
        chosen = np.random.choice(
            non_zero_indices, size=self.sample_size, replace=False
        )
        new_solution[chosen] = 1
        return new_solution

    def _mutate(self, solution):
        mutation_indices = np.random.choice(
            np.arange(len(solution)), size=2 * self.sample_size, replace=False
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

        return best_solution

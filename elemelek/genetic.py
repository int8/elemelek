import random

import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        similarity_matrix,
        target_similarity_median,
        population_size,
        k,
        max_gen,
        mutation_rate,
    ):
        self.similarity_matrix = similarity_matrix
        self.target_similarity_median = target_similarity_median
        self.population_size = population_size
        self.k = k
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate

    def _init_solution(self):
        solution = np.zeros(len(self.similarity_matrix), dtype=np.uint8)
        indices = random.sample(range(len(self.similarity_matrix)), self.k)
        solution[indices] = 1
        return solution

    def _eval(self, solution):
        submatrix = self.similarity_matrix[
            np.ix_(solution.nonzero()[0], solution.nonzero()[0])
        ]
        #         print(np.median(submatrix))
        return abs(np.median(submatrix) - self.target_similarity_median)

    def _cross(self, solution_a, solution_b):
        non_zero_indices = np.nonzero(solution_a + solution_b)[0]
        if len(non_zero_indices) < self.k:
            non_zero_indices = np.pad(
                non_zero_indices, (0, self.k - len(non_zero_indices)), "wrap"
            )
        new_solution = np.zeros(len(self.similarity_matrix), dtype=np.uint8)
        chosen = np.random.choice(non_zero_indices, size=self.k, replace=False)
        new_solution[chosen] = 1
        return new_solution

    def _mutate(self, solution):
        mutation_indices = np.random.choice(
            np.arange(len(solution)), size=2 * self.k, replace=False
        )
        for index in mutation_indices:
            solution[index] = 1 - solution[index]
        # Ensure exactly k entries are 1
        ones = np.where(solution == 1)[0]
        zeros = np.where(solution == 0)[0]
        if len(ones) > self.k:
            drop_indices = np.random.choice(
                ones, size=len(ones) - self.k, replace=False
            )
            solution[drop_indices] = 0
        elif len(ones) < self.k:
            add_indices = np.random.choice(
                zeros, size=self.k - len(ones), replace=False
            )
            solution[add_indices] = 1
        return solution

    def optimize(self):
        # Initialize population
        population = [self._init_solution() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = np.inf

        # Main loop
        for generation in range(self.max_gen):
            # Evaluate all solutions
            fitness = np.array([self._eval(sol) for sol in population])

            # Select best solution for showing progress
            min_index = np.argmin(fitness)
            #             print(fitness)
            #             print(min_index)
            if fitness[min_index] < best_fitness:
                best_fitness = fitness[min_index]
                best_solution = population[min_index]

            # Create new population with crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parents = random.sample(population, 2)
                offspring = self._cross(parents[0], parents[1])
                if random.random() < self.mutation_rate:
                    offspring = self._mutate(offspring)
                new_population.append(offspring)
            population = new_population

        return best_solution

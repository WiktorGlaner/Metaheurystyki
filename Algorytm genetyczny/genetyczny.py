import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random

# Dane problemu
weights = [
    32252, 225790, 468164, 489494, 35384, 265590, 497911, 800493, 823576, 
    552202, 323618, 382846, 44676, 169738, 610876, 854190, 671123, 698180, 
    446517, 909620, 904818, 730061, 931932, 952360, 926023, 978724
]
values = [
    68674, 471010, 944620, 962094, 78344, 579152, 902698, 1686515, 1688691, 
    1056157, 677562, 833132, 99192, 376418, 1253986, 1853562, 1320297, 1301637, 
    859835, 1677534, 1910501, 1528646, 1827477, 2068204, 1746556, 2100851
]
max_weight = 6404180

# Parametry bazowe
population_size = 50
num_generations = 100
crossover_prob = 0.8
mutation_prob = 0.05

# Funkcja przystosowania
def fitness(individual, weights, values, max_weight):
    total_weight = sum(ind * w for ind, w in zip(individual, weights))
    total_value = sum(ind * v for ind, v in zip(individual, values))
    return total_value if total_weight <= max_weight else 0

# Inicjalizacja populacji
def initialize_population(pop_size, num_items):
    return [np.random.randint(2, size=num_items).tolist() for _ in range(pop_size)]

# Selekcja - Koło ruletki
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.choice(population)
    selection_probs = [fit / total_fitness for fit in fitness_values]
    return population[np.random.choice(len(population), p=selection_probs)]

# Krzyżowanie jednopunktowe
def single_point_crossover(parent1, parent2, crossover_prob):
    if np.random.rand() < crossover_prob:
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

# Mutacja
def mutate(individual, mutation_prob):
    return [1 - gene if np.random.rand() < mutation_prob else gene for gene in individual]

# Algorytm genetyczny
def genetic_algorithm(weights, values, max_weight, pop_size, num_gens, crossover_prob, mutation_prob):
    num_items = len(weights)
    population = initialize_population(pop_size, num_items)
    best_solution = None
    best_fitness = 0
    fitness_history = []

    for gen in range(num_gens):
        fitness_values = [fitness(ind, weights, values, max_weight) for ind in population]
        new_population = []

        # Zachowaj najlepsze rozwiązanie
        max_gen_fitness = max(fitness_values)
        if max_gen_fitness > best_fitness:
            best_fitness = max_gen_fitness
            best_solution = population[np.argmax(fitness_values)]

        # Tworzenie nowej populacji
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = single_point_crossover(parent1, parent2, crossover_prob)
            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)
            new_population.extend([child1, child2])

        # Aktualizacja populacji
        population = new_population[:pop_size]
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history

# Funkcja do przeprowadzenia eksperymentów
def run_experiments(param_name, param_values, fixed_params):
    results = {}
    for value in param_values:
        all_runs = []
        for _ in range(5):  # Min. 5 razy dla każdego zestawu parametrów
            params = fixed_params.copy()
            params[param_name] = value
            _, best_fitness, fitness_history = genetic_algorithm(
                weights, values, max_weight, params['population_size'], params['num_generations'],
                params['crossover_prob'], params['mutation_prob']
            )
            all_runs.append(fitness_history)
        results[value] = all_runs
    return results

# Wykres wyników eksperymentów
def plot_experiment_results(results, title, xlabel, param_name):
    for value, runs in results.items():
        plt.figure(figsize=(14, 8))
        best_run = np.max(runs, axis=0)
        worst_run = np.min(runs, axis=0)
        avg_run = np.mean(runs, axis=0)

        plt.plot(best_run, label='Najlepszy wynik', linestyle='-', alpha=0.8)
        plt.plot(worst_run, label='Najgorszy wynik', linestyle='--', alpha=0.6)
        plt.plot(avg_run, label='Średni wynik', linestyle=':', alpha=0.8)

        plt.title(f'{title} (param={value})', fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('Najlepsze przystosowanie', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.2f} mln'))
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

# Eksperymenty
fixed_params = {
    'population_size': 50,
    'num_generations': 100,
    'crossover_prob': 0.8,
    'mutation_prob': 0.05
}

# 1. Wpływ pc
crossover_probs = [0.5, 0.8, 1.2]
results_pc = run_experiments('crossover_prob', crossover_probs, fixed_params)
plot_experiment_results(results_pc, 'Wpływ prawdopodobieństwa krzyżowania (pc)', 'Pokolenie', 'crossover_prob')

# 2. Wpływ pm
mutation_probs = [0.01, 0.05, 0.2]
results_pm = run_experiments('mutation_prob', mutation_probs, fixed_params)
plot_experiment_results(results_pm, 'Wpływ prawdopodobieństwa mutacji (pm)', 'Pokolenie', 'mutation_prob')

# 3. Wpływ rozmiaru populacji
population_sizes = [10, 50, 200]
results_pop_size = run_experiments('population_size', population_sizes, fixed_params)
plot_experiment_results(results_pop_size, 'Wpływ rozmiaru populacji (N)', 'Pokolenie', 'population_size')

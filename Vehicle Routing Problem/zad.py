import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import itertools
import pandas as pd

# Wczytywanie danych
class Customer:
    def __init__(self, no, x, y, demand, ready_time, due_date, service_time):
        self.no = no
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

def load_data(filename="R101.txt"):
    customers = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                parts = list(map(float, parts))
                customers.append(Customer(*parts))
            except ValueError:
                continue
    
    if len(customers) < 2:
        print("Błąd: Za mało klientów wczytanych z pliku!")
        exit(1)
    
    print(f"Wczytano {len(customers)} klientów.")
    return customers

# Generowanie macierzy odległości
customers = load_data()
distance_matrix = np.zeros((len(customers), len(customers)))
for i in range(len(customers)):
    for j in range(len(customers)):
        distance_matrix[i][j] = euclidean((customers[i].x, customers[i].y), (customers[j].x, customers[j].y))

# Klasa cząsteczki
class Particle:
    def __init__(self, num_customers, vehicle_capacity, lambda_penalty, inertia, c1, c2):
        self.position = np.random.permutation(num_customers).tolist()
        self.velocity = np.random.uniform(-2, 2, num_customers).tolist()
        self.best_position = self.position[:]
        self.best_fitness = float('inf')
        self.vehicles_used = 0
        self.routes = []
        self.vehicle_capacity = vehicle_capacity
        self.lambda_penalty = lambda_penalty
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.fitness = self.evaluate()

    def evaluate(self):
        total_distance = 0
        current_time = 0
        current_load = 0
        current_customer = 0
        vehicles_used = 1
        self.routes = [[]]
        
        for next_customer in self.position:
            travel_time = distance_matrix[current_customer][next_customer]
            arrival_time = current_time + travel_time
            if (
                current_load + customers[next_customer].demand > self.vehicle_capacity
                or arrival_time > customers[next_customer].due_date
            ):
                vehicles_used += 1
                current_load = 0
                current_time = 0
                current_customer = 0
                self.routes.append([])
                travel_time = distance_matrix[current_customer][next_customer]
                arrival_time = travel_time

            if arrival_time < customers[next_customer].ready_time:
                arrival_time = customers[next_customer].ready_time

            if arrival_time > customers[next_customer].due_date:
                self.vehicles_used = vehicles_used
                return float('inf')

            current_time = arrival_time + customers[next_customer].service_time
            current_load += customers[next_customer].demand
            total_distance += travel_time
            current_customer = next_customer
            self.routes[-1].append(next_customer)
        
        self.vehicles_used = vehicles_used
        return total_distance + self.lambda_penalty * vehicles_used

    def update_velocity(self, global_best_position):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (self.best_position[i] - self.position[i])
            social = self.c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = self.inertia * self.velocity[i] + cognitive + social
            self.velocity[i] = max(min(self.velocity[i], 2), -2)

    def update_position(self):
        swap_indices = random.sample(range(len(self.position)), 2)
        self.position[swap_indices[0]], self.position[swap_indices[1]] = (
            self.position[swap_indices[1]], self.position[swap_indices[0]])
        self.fitness = self.evaluate()
        if self.fitness < self.best_fitness:
            self.best_position = self.position[:]
            self.best_fitness = self.fitness

# Eksperymenty parami dla każdej możliwej kombinacji
paired_experiments = list(itertools.combinations([
    "inertia", "c1c2", "lambda_penalty", "vehicle_capacity"
], 2))

experiment_values = {
    "inertia": [0.01, 0.4, 0.7, 0.9],
    "c1c2": [(1.2, 1.2), (1.5, 1.5), (2.0, 2.0)],
    "lambda_penalty": [0.0001, 1, 50, 100, 200],
    "vehicle_capacity": [25, 50, 100, 150, 200]
}

results = []
for param1, param2 in paired_experiments:
    plt.figure(figsize=(10, 6))
    for value1 in experiment_values[param1]:
        for value2 in experiment_values[param2]:
            settings = {
                "inertia": 0.7, "c1": 1.5, "c2": 1.5, "vehicle_capacity": 100, "lambda_penalty": 1
            }
            settings[param1] = value1
            settings[param2] = value2
            particles = [Particle(len(customers) - 1, settings["vehicle_capacity"], settings["lambda_penalty"], settings["inertia"], settings["c1"], settings["c2"]) for _ in range(30)]
            global_best_particle = min(particles, key=lambda p: p.best_fitness)
            global_best_position = global_best_particle.best_position
            fitness_history = []
            for _ in range(100):
                for particle in particles:
                    particle.update_velocity(global_best_position)
                    particle.update_position()
                best_particle = min(particles, key=lambda p: p.best_fitness)
                global_best_particle = best_particle
                global_best_position = best_particle.best_position
                fitness_history.append(best_particle.best_fitness)
            results.append((param1, value1, param2, value2, min(fitness_history), global_best_particle.vehicles_used))
            plt.plot(fitness_history, label=f"{param1}={value1}, {param2}={value2}")
    plt.xlabel('Iteracja')
    plt.ylabel('Odległość')
    plt.title(f'Wpływ {param1} i {param2} na konwergencję')
    plt.legend()
    plt.show()

df_results = pd.DataFrame(results, columns=['Parametr1', 'Wartość1', 'Parametr2', 'Wartość2', 'Najlepsza odległość', 'Liczba pojazdów'])
print(df_results)

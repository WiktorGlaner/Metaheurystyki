import random
import numpy as np

class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_date, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

class VRPTW:
    def __init__(self, customers, vehicle_capacity, max_vehicles):
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity
        self.max_vehicles = max_vehicles
        self.depot = customers[0]  # Pierwszy klient to magazyn (depot)

    def calculate_distance(self, c1, c2):
        return np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    def create_initial_solution(self):
        # Tworzy losowe trasy, przestrzegając ograniczeń
        routes = []
        unassigned = self.customers[1:]  # Pomijamy magazyn
        random.shuffle(unassigned)

        while unassigned:
            route = [self.depot]
            capacity = 0
            current_time = 0

            while unassigned:
                next_customer = unassigned[0]
                if capacity + next_customer.demand > self.vehicle_capacity:
                    break

                # Sprawdź okno czasowe
                arrival_time = max(current_time + self.calculate_distance(route[-1], next_customer), next_customer.ready_time)
                if arrival_time > next_customer.due_date:
                    break

                route.append(next_customer)
                capacity += next_customer.demand
                current_time = arrival_time + next_customer.service_time
                unassigned.pop(0)

            route.append(self.depot)
            routes.append(route)

        return routes

    def fitness(self, routes):
        total_distance = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += self.calculate_distance(route[i], route[i + 1])
        return total_distance

    def crossover(self, parent1, parent2):
        # Krzyżowanie typu OX
        child = []
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child.extend(parent1[start:end])

        for customer in parent2:
            if customer not in child:
                child.append(customer)

        return child

    def mutate(self, routes):
        # Mutacja przez zamianę dwóch klientów
        for route in routes:
            if len(route) > 2:
                i, j = random.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]
        return routes

    def two_opt(self, route):
        # Poprawa trasy za pomocą 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if self.fitness([new_route]) < self.fitness([route]):
                        route = new_route
                        improved = True
        return route

    def solve(self, population_size=50, generations=100, mutation_rate=0.1):
        population = [self.create_initial_solution() for _ in range(population_size)]

        for generation in range(generations):
            population = sorted(population, key=lambda x: self.fitness(x))
            new_population = [population[0]]  # Elitizm

            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population[:10], k=2)  # Selekcja
                child = self.crossover(parent1, parent2)
                if random.random() < mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)

            population = new_population

        # Poprawa tras za pomocą 2-opt
        for i in range(len(population)):
            for j in range(len(population[i])):
                population[i][j] = self.two_opt(population[i][j])

        best_solution = min(population, key=lambda x: self.fitness(x))
        return best_solution, self.fitness(best_solution)

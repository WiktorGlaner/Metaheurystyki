import matplotlib.pyplot as plt
from vrptw_algorithm import VRPTW, Customer

def load_data(file_path):
    customers = []
    with open(file_path, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            customer = Customer(
                id=int(data[0]),
                x=data[1],
                y=data[2],
                demand=data[3],
                ready_time=data[4],
                due_date=data[5],
                service_time=data[6]
            )
            customers.append(customer)
    return customers

def run_experiments(file_path, population_size=50, generations=100, mutation_rate=0.1, runs=5):
    customers = load_data(file_path)
    results = []

    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        problem = VRPTW(customers, vehicle_capacity=200, max_vehicles=10)
        best_solution, best_fitness = problem.solve(population_size, generations, mutation_rate)
        results.append(best_fitness)
        print(f"Best fitness: {best_fitness}")

    return results

def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results, marker='o', linestyle='-', color='b')
    plt.title("Wyniki eksperymentów")
    plt.xlabel("Numer uruchomienia")
    plt.ylabel("Najlepsza wartość funkcji przystosowania")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "RC101.txt"
    results = run_experiments(file_path)
    plot_results(results)

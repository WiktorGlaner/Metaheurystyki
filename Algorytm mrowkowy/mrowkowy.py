import numpy as np
import matplotlib.pyplot as plt
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

#############################################
# Funkcje pomocnicze
#############################################

def load_data(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, x, y = parts
                coords.append((float(x), float(y)))
    return coords

def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            else:
                dist_matrix[i][j] = np.inf
    return dist_matrix

def path_length(path, dist_matrix):
    length = 0.0
    for i in range(len(path)-1):
        length += dist_matrix[path[i]][path[i+1]]
    return length

def build_solution(n, dist_matrix, tau, alpha, beta, random_choice_prob):
    start = random.randint(0, n-1)
    path = [start]
    visited = set(path)
    for _ in range(n-1):
        i = path[-1]
        candidates = [c for c in range(n) if c not in visited]
        if random.random() < random_choice_prob:
            next_city = random.choice(candidates)
        else:
            probs = []
            for c in candidates:
                tau_ij = tau[i][c]**alpha
                eta_ij = (1.0 / dist_matrix[i][c])**beta if dist_matrix[i][c] != 0 else 0
                probs.append(tau_ij * eta_ij)
            s = sum(probs)
            if s == 0:
                next_city = random.choice(candidates)
            else:
                r = random.random() * s
                sumownik = 0.0
                for idx, val in enumerate(probs):
                    sumownik += val
                    if sumownik >= r:
                        next_city = candidates[idx]
                        break
        path.append(next_city)
        visited.add(next_city)
    return path

def update_pheromones(tau, paths, dist_matrix, rho, Q):
    tau *= (1 - rho)
    for path, length in paths:
        delta = Q / length
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            tau[a][b] += delta
            tau[b][a] += delta

def run_aco(dist_matrix, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0, rho=0.5, Q=100, random_choice_prob=0.0):
    # Każdy proces będzie wykonywał tę funkcję.
    print(f"[PID {os.getpid()}] Uruchamiam run_aco z parametrami: num_ants={num_ants}, num_iterations={num_iterations}")
    n = dist_matrix.shape[0]
    tau = np.ones((n, n))
    best_path = None
    best_length = float('inf')

    for it in range(num_iterations):
        paths = []
        for k in range(num_ants):
            path = build_solution(n, dist_matrix, tau, alpha, beta, random_choice_prob)
            length = path_length(path, dist_matrix)
            paths.append((path, length))
            if length < best_length:
                best_length = length
                best_path = path
        update_pheromones(tau, paths, dist_matrix, rho, Q)

    return best_path, best_length

def plot_path(path, coords, title="Najlepsza trasa"):
    x = [coords[i][0] for i in path]
    y = [coords[i][1] for i in path]

    plt.figure(figsize=(8,8))
    plt.scatter(x, y, c='red')
    plt.plot(x, y, c='blue', linewidth=2)

    for i in range(len(path)):
        plt.text(x[i], y[i], str(path[i]), fontsize=9)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def experiment_param(dist_matrix, param_name, param_values, runs=5, fixed_params=None, coords=None):
    results = {}
    global_best_path = None
    global_best_length = float('inf')

    for pv in param_values:
        fp = fixed_params.copy()
        fp[param_name] = pv
        best_overall_path = None
        best_overall_length = float('inf')

        with ProcessPoolExecutor() as executor:
            futures = []
            for r in range(runs):
                print(f"Przy param {param_name}={pv}, run={r+1}/{runs}")
                futures.append(executor.submit(
                    run_aco,
                    dist_matrix,
                    fp['num_ants'],
                    fp['num_iterations'],
                    fp['alpha'],
                    fp['beta'],
                    fp['rho'],
                    fp['Q'],
                    fp['random_choice_prob']
                ))
            
            results_for_pv = []
            for f in as_completed(futures):
                path, length = f.result()
                results_for_pv.append(length)
                if length < best_overall_length:
                    best_overall_length = length
                    best_overall_path = path

        results[pv] = {
            'best': np.min(results_for_pv),
            'worst': np.max(results_for_pv),
            'mean': np.mean(results_for_pv),
            'all_runs': results_for_pv,
            'best_path': best_overall_path
        }

        # Aktualizujemy globalną najlepszą trasę jeśli ta dla pv jest lepsza
        if best_overall_length < global_best_length:
            global_best_length = best_overall_length
            global_best_path = best_overall_path

    # Po zakończeniu wszystkich wartości parametru pv, wyświetlamy najlepszą trasę spośród wszystkich
    if coords is not None and global_best_path is not None:
        plot_path(global_best_path, coords, title=f"Najlepsza trasa dla całego eksperymentu z parametrem {param_name}")

    return results

def plot_experiment(results, title, xlabel):
    param_values = sorted(results.keys())
    best_vals = [results[p]['best'] for p in param_values]
    worst_vals = [results[p]['worst'] for p in param_values]
    mean_vals = [results[p]['mean'] for p in param_values]

    plt.figure(figsize=(8,5))
    plt.plot(param_values, best_vals, label='Najlepszy wynik', marker='o')
    plt.plot(param_values, mean_vals, label='Średni wynik', marker='s')
    plt.plot(param_values, worst_vals, label='Najgorszy wynik', marker='^')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Długość trasy')
    plt.legend()
    plt.grid(True)
    plt.show()

#############################################
# Główna część kodu - przeprowadzanie eksperymentów
#############################################

if __name__ == "__main__":
    filename = "A-n80-k10.txt"  # Zmienić na właściwą ścieżkę do pliku z danymi
    coords = load_data(filename)
    dist_matrix = compute_distance_matrix(coords)

    # Stałe parametry bazowe
    fixed_params = {
        'num_ants': 20,
        'num_iterations': 50,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.5,
        'Q': 100,
        'random_choice_prob': 0.03
    }

    runs = 5  # Liczba powtórzeń dla każdego zestawu parametrów

    # Eksperyment 1: Wpływ liczebności mrówek
    ant_values = [5, 10, 20, 50]
    res_ants = experiment_param(dist_matrix, 'num_ants', ant_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_ants, 'Wpływ liczebności mrówek na jakość trasy', 'Liczba mrówek')

    # Przywrócenie wartości bazowych
    fixed_params['num_ants'] = 20

    # Eksperyment 2: Wpływ prawdopodobieństwa losowego wyboru
    random_prob_values = [0.01, 0.03, 0.05, 0.1]
    res_random = experiment_param(dist_matrix, 'random_choice_prob', random_prob_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_random, 'Wpływ prawdopodobieństwa losowego wyboru atrakcji', 'Prawdopodobieństwo wyboru losowego')

    fixed_params['random_choice_prob'] = 0.0

    # Eksperyment 3: Wpływ wagi feromonów (alpha)
    alpha_values = [0.1, 0.3, 0.8, 2]
    res_alpha = experiment_param(dist_matrix, 'alpha', alpha_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_alpha, 'Wpływ wagi feromonów (α)', 'α')

    fixed_params['alpha'] = 1.0

    # Eksperyment 4: Wpływ wagi heurystyki (beta)
    beta_values = [0.1, 0.3, 0.8, 2]
    res_beta = experiment_param(dist_matrix, 'beta', beta_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_beta, 'Wpływ wagi heurystyki (β)', 'β')

    fixed_params['beta'] = 2.0

    # Eksperyment 5: Wpływ liczby iteracji
    iter_values = [10, 25, 50, 100]
    res_iter = experiment_param(dist_matrix, 'num_iterations', iter_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_iter, 'Wpływ liczby iteracji', 'Liczba iteracji')

    fixed_params['num_iterations'] = 50

    # Eksperyment 6: Wpływ współczynnika wyparowywania (rho)
    rho_values = [0.05, 0.1, 0.3, 0.8]
    res_rho = experiment_param(dist_matrix, 'rho', rho_values, runs=runs, fixed_params=fixed_params, coords=coords)
    plot_experiment(res_rho, 'Wpływ współczynnika wyparowywania (rho)', 'rho')

    # Po wykonaniu eksperymentów prezentujemy przykładową trasę dla bazowych parametrów:
    best_path, best_length = run_aco(dist_matrix, 
                                     num_ants=fixed_params['num_ants'], 
                                     num_iterations=fixed_params['num_iterations'],
                                     alpha=fixed_params['alpha'],
                                     beta=fixed_params['beta'],
                                     rho=fixed_params['rho'],
                                     Q=fixed_params['Q'],
                                     random_choice_prob=fixed_params['random_choice_prob'])

    print("Najlepsza uzyskana trasa (dla bazowych parametrów):", best_path)
    print("Długość trasy:", best_length)
    plot_path(best_path, coords, title="Najlepsza znaleziona trasa dla bazowych parametrów")

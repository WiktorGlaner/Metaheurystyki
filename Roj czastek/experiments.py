from pso import PSO
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap


def run_experiment(func, bounds, num_particles_list, max_iter_list, w_values, c1c2_values, output_file):
    results = []
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Num_particles', 'Max_iter', 'w', 'c1', 'c2', 'Best_value'])

        for num_particles in num_particles_list:
            for max_iter in max_iter_list:
                for w in w_values:
                    for c1, c2 in c1c2_values:
                        # Optymalizacja z przechwyceniem historii
                        pso = PSO(func, bounds, num_particles, max_iter)
                        best_position, best_value, history = pso.optimize(w=w, c1=c1, c2=c2)

                        # Zapisanie wyników
                        results.append({
                            'num_particles': num_particles,
                            'max_iter': max_iter,
                            'w': w,
                            'c1': c1,
                            'c2': c2,
                            'best_value': best_value,
                            'history': history  # Dodanie historii
                        })
                        writer.writerow([num_particles, max_iter, w, c1, c2, best_value])

    return results



def plot_experiment_results(results):
    """
    Tworzy dwa rodzaje wykresów słupkowych:
    1. Grupowanie według liczby iteracji.
    2. Grupowanie według liczby cząstek.
    Etykiety osi X zawierają tylko parametry eksperymentu (w, c1, c2).
    """

    # Unikalne grupy liczby cząstek i liczby iteracji
    particle_groups = sorted(set(r['num_particles'] for r in results))
    iter_groups = sorted(set(len(r['history']) for r in results))

    # Kolory dla różnych grup
    cmap = get_cmap("tab10")
    particle_colors = {p: cmap(i / len(particle_groups)) for i, p in enumerate(particle_groups)}
    iter_colors = {i: cmap(j / len(iter_groups)) for j, i in enumerate(iter_groups)}

    # 1. Wykresy dla grup iteracji
    for num_iter in iter_groups:
        subset = [r for r in results if len(r['history']) == num_iter]
        labels = [f"(w={r['w']}, c1={r['c1']}, c2={r['c2']})" for r in subset]
        values = [r['best_value'] for r in subset]
        colors = [particle_colors[r['num_particles']] for r in subset]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(values)), values, color=colors)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=10)
        plt.xlabel('Parametry (w, c1, c2)')
        plt.ylabel('Najlepsza wartość globalna')
        plt.title(f'Porównanie wyników dla liczby iteracji: {num_iter}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color=particle_colors[p], lw=4, label=f"{p} cząstek")
                for p in particle_groups
            ],
            title="Liczba cząstek"
        )
        plt.tight_layout()
        plt.show()

    # 2. Wykresy dla grup liczby cząstek
    for particles in particle_groups:
        subset = [r for r in results if r['num_particles'] == particles]
        labels = [f"(w={r['w']}, c1={r['c1']}, c2={r['c2']})" for r in subset]
        values = [r['best_value'] for r in subset]
        colors = [iter_colors[len(r['history'])] for r in subset]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(values)), values, color=colors)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=10)
        plt.xlabel('Parametry (w, c1, c2)')
        plt.ylabel('Najlepsza wartość globalna')
        plt.title(f'Porównanie wyników dla liczby cząstek: {particles}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color=iter_colors[i], lw=4, label=f"{i} iteracji")
                for i in iter_groups
            ],
            title="Liczba iteracji"
        )
        plt.tight_layout()
        plt.show()

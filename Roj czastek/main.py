from pso import PSO
from functions import rastrigin, ackley
from visualizations import plot_3d_function, plot_cross_sections, plot_convergence
from experiments import run_experiment, plot_experiment_results

def main():
    bounds = [-5, 5]
    print("Wybierz tryb pracy:")
    print("1: Przeprowadzenie eksperymentów")
    print("2: Znalezienie minimum globalnego")
    choice = input("Wybierz opcję (1 lub 2): ")

    if choice == '1':
        print("Przeprowadzanie eksperymentów...")
        num_particles_list = [10, 20, 30, 40]
        max_iter_list = [50, 100, 200]
        w_values = [0.3, 0.5, 0.7]
        c1c2_values = [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]

        func = rastrigin  # Można zmienić na ackley
        output_file = 'experiment_results.csv'
        results = run_experiment(func, bounds, num_particles_list, max_iter_list, w_values, c1c2_values, output_file)

        print("Eksperyment zakończony. Wyniki zapisano w pliku.")
        plot_experiment_results(results)

    elif choice == '2':
        print("Obliczanie minimum globalnego...")
        func_choice = input("Wybierz funkcję: 1 - Rastrigin, 2 - Ackley: ")
        func = rastrigin if func_choice == '1' else ackley

        num_particles = int(input("Podaj liczbę cząstek: "))
        max_iter = int(input("Podaj maksymalną liczbę iteracji: "))
        w = float(input("Podaj wartość wagi bezwładności (w): "))
        c1 = float(input("Podaj wartość współczynnika c1: "))
        c2 = float(input("Podaj wartość współczynnika c2: "))

        pso = PSO(func, bounds, num_particles, max_iter)
        best_position, best_value, history = pso.optimize(w=w, c1=c1, c2=c2)

        print(f"Najlepsza pozycja: {best_position}, Najlepsza wartość: {best_value}")
        plot_3d_function(func, bounds, "Funkcja z minimum globalnym", best_position)
        plot_cross_sections(func, bounds, best_position)
        plot_convergence(history, "Zbieżność algorytmu")

if __name__ == "__main__":
    main()

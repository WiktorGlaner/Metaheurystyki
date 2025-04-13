import numpy as np

def funkcja_z_rozdzialu_3(x):
    if -105 < x < -95:
        return 10 - 2 * abs(x + 100)
    elif 95 < x < 105:
        return 11 - 2.2 * abs(x - 100)
    else:
        return 0

def funkcja_z_rozdzialu_4(x):
    return x * np.sin(10 * np.pi * x) + 1

def symulowane_wyzarzanie(f, przedzial, T_poczatkowe, alfa, M, k):
    s = np.random.uniform(przedzial[0], przedzial[1])
    s_best = s
    T = T_poczatkowe
    max_step = przedzial[1] - przedzial[0]

    for i in range(M):
        # Generowanie nowego rozwiązania
        step_size = max_step * (T / T_poczatkowe)
        s_prime = s + np.random.uniform(-step_size, step_size)
        s_prime = np.clip(s_prime, przedzial[0], przedzial[1])

        # Obliczenie różnicy energii (absolutne zmiany wartości)
        delta_E = abs(f(s_prime)) - abs(f(s))

        # Akceptacja nowego rozwiązania
        if delta_E > 0:
            s = s_prime
        else:
            #mechanizm Metropolisa
            p = np.exp(delta_E / (k * T))
            if np.random.rand() < p:
                s = s_prime

        # Zapamiętanie najlepszego ekstremum
        if abs(f(s)) > abs(f(s_best)):
            s_best = s

        # Aktualizacja temperatury
        T = alfa * T

    return s_best, f(s_best)

# Parametry zgodne z artykułem
przedzial_3 = [-150, 150]
T_poczatkowe_3 = 500
alfa_3 = 0.999
M_3 = 3000
k_3 = 0.1


# Uruchomienie algorytmu dla funkcji z rozdziału 3
s_best_3, f_best_3 = symulowane_wyzarzanie(funkcja_z_rozdzialu_3, przedzial_3, T_poczatkowe_3, alfa_3, M_3, k_3)
print(f"Najlepsze znalezione rozwiązanie dla funkcji z rozdziału 3: x = {s_best_3}, f(x) = {f_best_3}")

# Parametry początkowe dla funkcji z rozdziału 4
przedzial_4 = [-1, 2]
T_poczatkowe_4 = 5
alfa_4 = 0.997
M_4 = 1200
k_4 = 0.1

# Uruchomienie algorytmu dla funkcji z rozdziału 4
s_best_4, f_best_4 = symulowane_wyzarzanie(funkcja_z_rozdzialu_4, przedzial_4, T_poczatkowe_4, alfa_4, M_4, k_4)
print(f"Najlepsze znalezione rozwiązanie dla funkcji z rozdziału 4: x = {s_best_4}, f(x) = {f_best_4}")

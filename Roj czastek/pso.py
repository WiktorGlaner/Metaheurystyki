import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.value = float('inf')

    def update_velocity(self, global_best, w, c1, c2):
        r1, r2 = np.random.random(), np.random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

class PSO:
    def __init__(self, func, bounds, num_particles, max_iter):
        self.func = func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], 2)
        self.global_best_value = float('inf')
        self.particles = [Particle(bounds) for _ in range(num_particles)]

    def optimize(self, w=0.5, c1=1.5, c2=1.5):
        history = []  # Historia najlepszych wartości globalnych
        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.value = self.func(*particle.position)
                if particle.value < particle.best_value:
                    particle.best_value = particle.value
                    particle.best_position = particle.position

                if particle.value < self.global_best_value:
                    self.global_best_value = particle.value
                    self.global_best_position = particle.position

            history.append(self.global_best_value)  # Zapisanie najlepszej wartości w tej iteracji

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position(self.bounds)

            #print(f"Iteracja {iteration+1}: Najlepsza wartość globalna = {self.global_best_value}")

        return self.global_best_position, self.global_best_value, history


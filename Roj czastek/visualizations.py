import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def plot_3d_function(func, bounds, title, best_position=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import numpy as np

    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = [ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)]
    fig.colorbar(surface[0], ax=ax, shrink=0.5, aspect=10, label='Wartość funkcji')

    if best_position is not None:
        z_min = func(best_position[0], best_position[1])
        # Linie przecinające się w punkcie minimum
        ax.plot([best_position[0], best_position[0]], [best_position[1], best_position[1]], [bounds[0], z_min],
                color='red', linestyle='--', label='Linia Z')
        ax.plot([bounds[0], best_position[0]], [best_position[1], best_position[1]], [z_min, z_min],
                color='blue', linestyle='--', label='Linia X')
        ax.plot([best_position[0], best_position[0]], [bounds[0], best_position[1]], [z_min, z_min],
                color='green', linestyle='--', label='Linia Y')
        ax.scatter(best_position[0], best_position[1], z_min, color='black', s=100, label='Globalne minimum')
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Dodanie suwaków
    ax_slider_x = plt.axes([0.25, 0.06, 0.65, 0.03])
    ax_slider_y = plt.axes([0.25, 0.03, 0.65, 0.03])
    ax_slider_z = plt.axes([0.25, 0.00, 0.65, 0.03])

    slider_x = Slider(ax_slider_x, 'X Max', bounds[0], bounds[1], valinit=bounds[1])
    slider_y = Slider(ax_slider_y, 'Y Max', bounds[0], bounds[1], valinit=bounds[1])
    slider_z = Slider(ax_slider_z, 'Z Max', Z.min(), Z.max(), valinit=Z.max())

    def update(val):
        x_limit = slider_x.val
        y_limit = slider_y.val
        z_limit = slider_z.val

        surface[0].remove()  # Usunięcie starej powierzchni
        mask_x = (X <= x_limit)
        mask_y = (Y <= y_limit)
        mask = mask_x & mask_y
        Z_masked = np.where(mask, Z, np.nan)
        surface[0] = ax.plot_surface(X, Y, np.minimum(Z_masked, z_limit), cmap='viridis', edgecolor='none', alpha=0.8)
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()


def plot_cross_sections(func, bounds, best_position):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)

    # Przekrój wzdłuż Y
    z_cross_y = func(x, best_position[1])
    plt.figure(figsize=(10, 6))
    plt.plot(x, z_cross_y, label=f'Przekrój przy y={best_position[1]:.2f}')
    plt.scatter([best_position[0]], [func(best_position[0], best_position[1])], color='red', label='Globalne minimum')
    plt.title("Przekrój wzdłuż osi Y", fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Przekrój wzdłuż X
    z_cross_x = func(best_position[0], y)
    plt.figure(figsize=(10, 6))
    plt.plot(y, z_cross_x, label=f'Przekrój przy x={best_position[0]:.2f}')
    plt.scatter([best_position[1]], [func(best_position[0], best_position[1])], color='red', label='Globalne minimum')
    plt.title("Przekrój wzdłuż osi X", fontsize=14)
    plt.xlabel('Y', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

def plot_convergence(history, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Najlepsza wartość globalna')
    plt.xlabel('Iteracja', fontsize=12)
    plt.ylabel('Najlepsza wartość', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


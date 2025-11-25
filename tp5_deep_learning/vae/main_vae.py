import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import sys

# Agregar el directorio actual y el padre al path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vae import VariationalAutoencoder


def load_icons(icons_dir='../resources/icons'):
    icon_files = sorted([f for f in os.listdir(icons_dir) if f.endswith('.png')])
    icon_images = []
    X = []
    
    for icon_file in icon_files:
        icon_path = os.path.join(icons_dir, icon_file)
        img = Image.open(icon_path).convert('L')  # Convertir a escala de grises
        icon_images.append(img)
        
        # Convertir a array y normalizar a [0, 1]
        img_array = np.array(img).flatten() / 255.0
        X.append(img_array)
    
    X = np.array(X)
    image_shape = icon_images[0].size  # (width, height)
    
    print(f"Cargados {len(icon_files)} iconos de tamaño {image_shape}")
    print(f"Forma de los datos: {X.shape}")
    
    return X, icon_images, image_shape #Retorna matriz X con todas las imágenes


def add_noise(X, noise_std=0.3):
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = np.clip(X + noise, 0, 1)
    return X_noisy


def visualize_original_icons(X, image_shape, results_dir='results'):
    """
    Visualiza todos los iconos originales antes del entrenamiento.
    """
    n_icons = X.shape[0]
    n_cols = min(8, n_icons)
    n_rows = (n_icons + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_icons):
        row = i // n_cols
        col = i % n_cols
        
        img = X[i].reshape(image_shape[1], image_shape[0])
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Icon {i+1}', fontsize=10)
    
    # Ocultar ejes sobrantes
    for i in range(n_icons, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'original_icons.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Iconos originales guardados en '{output_path}'")
    plt.close()


def visualize_reconstructions(vae, X, image_shape, results_dir='results', n_samples=8):
    n_samples = min(n_samples, X.shape[0])
    X_sample = X[:n_samples]
    X_recon = vae.reconstruct(X_sample)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        # Original
        img_orig = X_sample[i].reshape(image_shape[1], image_shape[0])
        axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstrucción
        img_recon = X_recon[i].reshape(image_shape[1], image_shape[0])
        axes[1, i].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstrucción', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_reconstructions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Reconstrucciones guardadas en '{output_path}'")
    plt.close()  # plt.show()


def visualize_denoising(vae, X, image_shape, results_dir='results', noise_std=0.3, n_samples=8):
    n_samples = min(n_samples, X.shape[0])
    X_sample = X[:n_samples]
    
    # Agregar ruido
    X_noisy = add_noise(X_sample, noise_std)
    
    # Reconstruir desde versión con ruido
    X_recon = vae.reconstruct(X_noisy)
    
    # Visualizar: Original limpio | Con ruido | Reconstruido
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
    
    for i in range(n_samples):
        # Original limpio
        img_orig = X_sample[i].reshape(image_shape[1], image_shape[0])
        axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Con ruido
        img_noisy = X_noisy[i].reshape(image_shape[1], image_shape[0])
        axes[1, i].imshow(img_noisy, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f'Con ruido (σ={noise_std})', fontsize=10)
        
        # Reconstruido
        img_recon = X_recon[i].reshape(image_shape[1], image_shape[0])
        axes[2, i].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstruido', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'vae_denoising_std_{noise_std}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Denoising guardado en '{output_path}'")
    plt.close()  # plt.show()


def visualize_latent_space(vae, X, results_dir='results'):
    """
    Visualiza la distribución de los datos en el espacio latente 2D.
    """
    mu = vae.get_latent_representation(X) #devuelve la media para cada icono 
     
    plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], s=100, alpha=0.7, edgecolors='black')
    
    # Etiquetar cada punto con su índice
    for i in range(len(mu)):
        plt.annotate(f'Icon {i+1}', (mu[i, 0], mu[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Dimensión latente 1')
    plt.ylabel('Dimensión latente 2')
    plt.title('Espacio latente del VAE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_latent_space.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Espacio latente guardado en '{output_path}'")
    plt.close()  # plt.show()


def visualize_manifold(vae, image_shape, results_dir='results', n_grid=15):
    """
    Visualiza el manifold muestreando un grid del espacio latente 2D.
    Cada punto del grid se decodifica para ver qué genera el VAE.
    """
    # Crear un grid en el espacio latente
    # Usar un rango que cubra la distribución normal estándar
    grid_range = 3  # ±3 desviaciones estándar cubre ~99.7% de N(0,1)
    grid_x = np.linspace(-grid_range, grid_range, n_grid)
    grid_y = np.linspace(-grid_range, grid_range, n_grid)
    
    # Crear figura
    figure = np.zeros((image_shape[1] * n_grid, image_shape[0] * n_grid))
    
    # Para cada punto del grid, decodificar
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Punto en el espacio latente
            z_sample = np.array([[xi, yi]])
            
            # Decodificar usando el método decode del VAE
            x_decoded, _ = vae.decode(z_sample)
            
            # Reshape y colocar en la figura
            digit = x_decoded[0].reshape(image_shape[1], image_shape[0])
            figure[i * image_shape[1]: (i + 1) * image_shape[1],
                   j * image_shape[0]: (j + 1) * image_shape[0]] = digit
    
    # Visualizar
    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Manifold del VAE - Grid {n_grid}x{n_grid} en espacio latente', fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_manifold.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Manifold guardado en '{output_path}'")
    plt.close()  # plt.show()


def generate_new_samples(vae, image_shape, results_dir='results', n_samples=8):
    """
    Genera nuevas muestras desde la distribución previa.
    """
    X_generated = vae.generate(n_samples)
    
    n_cols = min(8, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        
        img = X_generated[i].reshape(image_shape[1], image_shape[0])
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Generado {i+1}', fontsize=10)
    
    # Ocultar ejes sobrantes
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_generated_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Muestras generadas guardadas en '{output_path}'")
    plt.close()  # plt.show()


def interpolate_between_icons(vae, X, image_shape, results_dir='results', idx1=0, idx2=1, n_steps=10):
    """
    Interpola entre dos iconos en el espacio latente.
    """
    X1 = X[idx1]
    X2 = X[idx2]
    
    X_interp = vae.interpolate(X1, X2, n_steps)
    
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 2))
    
    for i in range(n_steps):
        img = X_interp[i].reshape(image_shape[1], image_shape[0])
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f'Icon {idx1+1}', fontsize=9)
        elif i == n_steps - 1:
            axes[i].set_title(f'Icon {idx2+1}', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'vae_interpolation_{idx1+1}_to_{idx2+1}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Interpolación guardada en '{output_path}'")
    plt.close()  # plt.show()


def visualize_latent_grid(vae, X_original, image_shape, results_dir='results', grid_size=15, x_range=(-3, 3), y_range=(-3, 3), name='vae_latent_grid'):
    """
    Visualiza una grilla de muestras generadas explorando el espacio latente 2D.
    Marca los puntos correspondientes a los íconos originales.

    Args:
        vae: Modelo VAE entrenado
        X_original: Datos originales para obtener sus representaciones latentes
        image_shape: Forma de las imágenes (width, height)
        results_dir: Directorio para guardar resultados
        grid_size: Tamaño de la grilla
        x_range: Rango para z₁
        y_range: Rango para z₂
        name: Nombre del archivo de salida
    """
    if vae.latent_dim != 2:
        print(f"  La visualización de grilla requiere espacio latente 2D (actual: {vae.latent_dim}D)")
        return

    # Obtener las representaciones latentes de los íconos originales
    mu_original = vae.get_latent_representation(X_original)

    # Generar grilla (ya viene ordenada por filas y columnas)
    X_grid = vae.generate_grid(x_samples=grid_size, y_samples=grid_size,
                               x_range=x_range, y_range=y_range)

    # Valores de las dimensiones latentes
    x_values = np.linspace(x_range[0], x_range[1], grid_size)
    y_values = np.linspace(y_range[1], y_range[0], grid_size)  # De max a min (top to bottom)

    # Crear figura con GridSpec para mejor control del layout
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(grid_size + 2, grid_size + 2, figure=fig,
                  left=0.08, right=0.98, top=0.95, bottom=0.05,
                  hspace=0.02, wspace=0.02)

    # Crear grilla de imágenes (dejando espacio para ejes)
    # Guardar referencias a los axes para agregar marcadores después
    axes_grid = {}  # (i, j) -> ax

    for i in range(grid_size):  # i = fila (índice y)
        for j in range(grid_size):  # j = columna (índice x)
            ax = fig.add_subplot(gs[i+1, j+1])  # +1 para dejar espacio para labels

            idx = i * grid_size + j  # Índice en el array X_grid
            img = X_grid[idx].reshape(image_shape[1], image_shape[0])
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Guardar referencia al axes
            axes_grid[(i, j)] = ax

    # Agregar etiquetas del eje X (z₁) arriba
    for j in range(grid_size):
        ax = fig.add_subplot(gs[0, j+1])
        ax.text(0.5, 0.5, f'{x_values[j]:.1f}',
               ha='center', va='center', fontsize=9, weight='bold')
        ax.axis('off')

    # Agregar etiquetas del eje Y (z₂) a la izquierda
    for i in range(grid_size):
        ax = fig.add_subplot(gs[i+1, 0])
        ax.text(0.5, 0.5, f'{y_values[i]:.1f}',
               ha='center', va='center', fontsize=9, weight='bold')
        ax.axis('off')

    # Título del eje X
    ax_title_x = fig.add_subplot(gs[grid_size+1, 1:grid_size+1])
    ax_title_x.text(0.5, 0.5, 'z₁ (Dimensión latente 1)',
                   ha='center', va='center', fontsize=13, weight='bold')
    ax_title_x.axis('off')

    # Título del eje Y (vertical)
    ax_title_y = fig.add_subplot(gs[1:grid_size+1, grid_size+1])
    ax_title_y.text(0.5, 0.5, 'z₂ (Dimensión latente 2)',
                   ha='center', va='center', fontsize=13, weight='bold',
                   rotation=-90)
    ax_title_y.axis('off')

    # Marcar los puntos correspondientes a los íconos originales
    # Encontrar las celdas más cercanas a cada punto latente original
    # Crear un diccionario para agrupar íconos que caen en la misma celda
    cell_icons = {}  # (i, j) -> [lista de índices de íconos]

    for icon_idx in range(len(mu_original)):
        z_point = mu_original[icon_idx]  # (z1, z2)

        # Encontrar la celda más cercana en la grilla
        dist_x = np.abs(x_values - z_point[0])
        dist_y = np.abs(y_values - z_point[1])

        # Índices de la celda más cercana
        j_closest = np.argmin(dist_x)  # columna
        i_closest = np.argmin(dist_y)  # fila

        # Agregar al diccionario
        cell_key = (i_closest, j_closest)
        if cell_key not in cell_icons:
            cell_icons[cell_key] = []
        cell_icons[cell_key].append(icon_idx)

    # Marcar las celdas con los íconos
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    for (i_cell, j_cell), icon_list in cell_icons.items():
        # Obtener el axes correspondiente a esta celda (ya existente)
        ax = axes_grid[(i_cell, j_cell)]

        # Si hay un solo ícono, usar su color específico
        # Si hay múltiples, usar un color mixto o mostrar todos
        if len(icon_list) == 1:
            color = colors[icon_list[0] % len(colors)]
            label_text = f'{icon_list[0]+1}'
        else:
            # Múltiples íconos en la misma celda
            color = 'black'
            label_text = ','.join([str(idx+1) for idx in icon_list])

        # Agregar un rectángulo de borde alrededor de la imagen
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 1, 1, linewidth=4, edgecolor=color,
                        facecolor='none', transform=ax.transAxes, zorder=10)
        ax.add_patch(rect)

        # Agregar etiqueta dentro de la imagen (esquina superior izquierda)
        # Posición dentro del axes para que sea visible
        ax.text(0.15, 0.85, label_text, transform=ax.transAxes,
               ha='center', va='center',
               fontsize=10 if len(label_text) < 4 else 8,
               weight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                        edgecolor='black', linewidth=2, alpha=0.95),
               zorder=11)

    # Título principal
    fig.suptitle('Grilla de muestras generadas en el espacio latente del VAE',
                fontsize=16, weight='bold', y=0.98)

    output_path = os.path.join(results_dir, f"{name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grilla del espacio latente guardada en '{output_path}'")
    print(f"  Íconos originales marcados en la grilla:")
    for (i, j), icons in sorted(cell_icons.items()):
        z_approx = (x_values[j], y_values[i])
        icon_str = ', '.join([f'{idx+1}' for idx in icons])
        print(f"    Celda [{i},{j}] (z≈{z_approx[0]:.1f}, {z_approx[1]:.1f}): íconos {icon_str}")
    plt.show()


def plot_training_history(vae, results_dir='results'):
    """
    Grafica la evolución de la pérdida durante el entrenamiento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(vae.loss_history, label='Pérdida total')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.title('Evolución del entrenamiento del VAE')
    plt.ylim(0, 5)  # Fijar el rango del eje y entre 0 y 5
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Historial de entrenamiento guardado en '{output_path}'")
    plt.close()  # plt.show()

def main():
    # Obtener el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar configuración
    if len(sys.argv) < 2:
        config_path = os.path.join(script_dir, 'config.json')
    else:
        config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Crear directorio de resultados si no existe
    results_dir = os.path.join(script_dir, config.get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    # Cargar iconos
    print("=" * 60)
    print("CARGANDO DATOS")
    print("=" * 60)
    icons_path = os.path.join(script_dir, config.get('icons_dir', '../resources/icons'))
    X, icon_images, image_shape = load_icons(icons_path)
    input_dim = X.shape[1]
    
    # Configurar arquitectura del VAE
    latent_dim = config.get('vae_latent_dim', 2)
    h1 = config.get('vae_h1', 64)
    h2 = config.get('vae_h2', None)
    h3 = config.get('vae_h3', None)
    h4 = config.get('vae_h4', None)

    if h2 is None:
        encoder_layers = [input_dim, h1, latent_dim]
        decoder_layers = [latent_dim, h1, input_dim]
    elif h3 is None:
        encoder_layers = [input_dim, h1, h2, latent_dim]
        decoder_layers = [latent_dim, h2, h1, input_dim]
    elif h4 is None:
        encoder_layers = [input_dim, h1, h2, h3, latent_dim]
        decoder_layers = [latent_dim, h3, h2, h1, input_dim]
    else:
        encoder_layers = [input_dim, h1, h2, h3, h4, latent_dim]
        decoder_layers = [latent_dim, h4, h3, h2, h1, input_dim]

    print(f"\nArquitectura del VAE:")
    print(f"  Encoder: {encoder_layers}")
    print(f"  Decoder: {decoder_layers}")
    
    # Crear VAE
    vae = VariationalAutoencoder(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        activation=config.get('activation', 'tanh'),
        eta=config.get('vae_learning_rate', 0.005),
        optimizer=config.get('optimizer', 'adam'),
        batch_size=config.get('vae_batch_size', None),
        seed=config.get('seed', 42),
        kl_weight=config.get('vae_beta', 1.0)
    )
    
    # Entrenar VAE
    print("\n" + "=" * 60)
    print("ENTRENANDO VAE")
    print("=" * 60)
    vae.train(
        X, 
        epochs=config.get('vae_epochs', 3000),
        epsilon=config.get('epsilon', 1e-6),
        verbose=True
    )
    
    # Visualizaciones
    print("\n" + "=" * 60)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 60)
    
    print("\n0. Iconos originales...")
    visualize_original_icons(X, image_shape, results_dir)
    
    print("\n1. Historial de entrenamiento...")
    plot_training_history(vae, results_dir)
    
    print("\n2. Reconstrucciones...")
    visualize_reconstructions(vae, X, image_shape, results_dir, n_samples=min(8, len(X)))
    
    print("\n3. Denoising (Capacidad Generativa)...")
    # Probar con diferentes niveles de ruido
    noise_levels = config.get('noise_levels', [0.2, 0.8, 0.9])
    for noise_std in noise_levels:
        print(f"   - Ruido σ={noise_std}")
        visualize_denoising(vae, X, image_shape, results_dir, noise_std=noise_std, n_samples=min(8, len(X)))
    
    print("\n4. Espacio latente...")
    if latent_dim == 2:
        visualize_latent_space(vae, X, results_dir)
        print("\n4b. Manifold del espacio latente...")
        visualize_manifold(vae, image_shape, results_dir, n_grid=15)
    else:
        print(f"  Espacio latente tiene dimensión {latent_dim}, se necesita dimensión 2 para visualizar")
    
    print("\n5. Grilla del espacio latente...")
    if latent_dim == 2:
        grid_size = 10
        x_range = (-2, 2)
        y_range = (-2, 2)
        visualize_latent_grid(vae, X, image_shape, results_dir, grid_size=grid_size,
                            x_range=x_range, y_range=y_range)
    else:
        print(f"  La grilla requiere espacio latente 2D (actual: {latent_dim}D)")

    print("\n6. Grilla del espacio latente...")
    if latent_dim == 2:
        grid_size = 15
        x_range = (-5, 5)
        y_range = (-5, 5)
        visualize_latent_grid(vae, X, image_shape, results_dir, grid_size=grid_size,
                              x_range=x_range, y_range=y_range, name='vae_latent_grid_wide')
    else:
        print(f"  La grilla requiere espacio latente 2D (actual: {latent_dim}D)")

    print("\n6. Muestras generadas...")
    generate_new_samples(vae, image_shape, results_dir, n_samples=8)
    
    print("\n7. Interpolaciones...")
    if len(X) >= 2:
        interpolate_between_icons(vae, X, image_shape, results_dir, idx1=0, idx2=1, n_steps=10)
        if len(X) >= 4:
            interpolate_between_icons(vae, X, image_shape, results_dir, idx1=2, idx2=3, n_steps=10)
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO Y VISUALIZACIONES COMPLETADOS")
    print("=" * 60)
    print(f"\nÉpocas de convergencia: {vae.converged_epoch}")
    print(f"Pérdida final: {vae.loss_history[-1]:.6f}")


if __name__ == "__main__":
    main()

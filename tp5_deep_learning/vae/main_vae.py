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
    
    return X, icon_images, image_shape


def add_noise(X, noise_std=0.3):
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = np.clip(X + noise, 0, 1)
    return X_noisy


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
    plt.show()


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
    plt.show()


def visualize_latent_space(vae, X, results_dir='results'):
    """
    Visualiza la distribución de los datos en el espacio latente 2D.
    """
    mu = vae.get_latent_representation(X)
    
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
    plt.show()


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
    plt.show()


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
    plt.show()


def plot_training_history(vae, results_dir='results'):
    """
    Grafica la evolución de la pérdida durante el entrenamiento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(vae.loss_history, label='Pérdida total')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Evolución del entrenamiento del VAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'vae_training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Historial de entrenamiento guardado en '{output_path}'")
    plt.show()


def main():
    # Obtener el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar configuración
    config_path = os.path.join(script_dir, 'config.json')
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
    h2 = config.get('vae_h2', 32)
    
    encoder_layers = [input_dim, h1, h2, latent_dim]
    decoder_layers = [latent_dim, h2, h1, input_dim]
    
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
    
    print("\n1. Historial de entrenamiento...")
    plot_training_history(vae, results_dir)
    
    print("\n2. Reconstrucciones...")
    visualize_reconstructions(vae, X, image_shape, results_dir, n_samples=min(8, len(X)))
    
    print("\n3. Denoising (Capacidad Generativa)...")
    # Probar con diferentes niveles de ruido
    noise_levels = config.get('noise_levels', [0.2, 0.4, 0.6])
    for noise_std in noise_levels:
        print(f"   - Ruido σ={noise_std}")
        visualize_denoising(vae, X, image_shape, results_dir, noise_std=noise_std, n_samples=min(8, len(X)))
    
    print("\n4. Espacio latente...")
    if latent_dim == 2:
        visualize_latent_space(vae, X, results_dir)
    else:
        print(f"  Espacio latente tiene dimensión {latent_dim}, se necesita dimensión 2 para visualizar")
    
    print("\n5. Muestras generadas...")
    generate_new_samples(vae, image_shape, results_dir, n_samples=8)
    
    print("\n6. Interpolaciones...")
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

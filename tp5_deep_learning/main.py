import numpy as np
import matplotlib.pyplot as plt
import json
import os
from seaborn import heatmap
from font import font_3, to_bin_array, monocromatic_cmap
from autoencoder import Autoencoder

def prepare_data():
    """Prepara los datos del font para el autoencoder"""
    # Convertir cada carácter a array binario y aplanarlo
    data = []
    for caracter in font_3:
        bin_array = to_bin_array(caracter)
        flattened = bin_array.flatten()  # 7x5 = 35 valores
        data.append(flattened)
    
    # Convertir a numpy array y normalizar a [0, 1]
    X = np.array(data, dtype=np.float32)
    return X

def plot_reconstructions(autoencoder, X, original_chars, n_samples=8):
    """Visualiza reconstrucciones del autoencoder"""
    # Seleccionar algunos caracteres aleatorios
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i, idx in enumerate(indices):
        original = X[idx].reshape(7, 5)
        reconstructed = autoencoder.reconstruct(X[idx:idx+1])[0].reshape(7, 5)
        
        # Original
        axes[0, i].imshow(original, cmap='binary', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')
        
        # Reconstrucción
        axes[1, i].imshow(reconstructed, cmap='binary', vmin=0, vmax=1)
        axes[1, i].set_title(f'Reconstruido')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/reconstructions.png', dpi=150, bbox_inches='tight')
    print("Reconstrucciones guardadas en 'reconstructions.png'")
    plt.show()

def plot_all_reconstructions(autoencoder, X, original_chars):
    """Visualiza todos los caracteres del dataset: originales y reconstruidos"""
    n_chars = len(X)
    
    # Obtener caracteres ASCII correspondientes
    char_labels = []
    for i in range(n_chars):
        ascii_val = 0x60 + i
        if ascii_val < 0x7f:
            char_labels.append(chr(ascii_val))
        else:
            char_labels.append('DEL')
    
    # Organizar en grilla: 4 filas x 8 columnas (32 caracteres)
    n_cols = 8
    n_rows = 4
    
    # Crear figura: 4 filas para originales, 4 filas para reconstruidos
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(20, 10))
    
    # Primera sección (4 filas): Originales
    for i in range(n_chars):
        row = i // n_cols  # Fila dentro de la sección de originales (0-3)
        col = i % n_cols
        
        original = X[i].reshape(7, 5)
        axes[row, col].imshow(original, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_title(f"'{char_labels[i]}'", fontsize=9, fontweight='bold')
        axes[row, col].axis('off')
    
    # Segunda sección (4 filas): Reconstruidos
    for i in range(n_chars):
        row = (i // n_cols) + n_rows  # Fila en la sección de reconstruidos (4-7)
        col = i % n_cols
        
        reconstructed = autoencoder.reconstruct(X[i:i+1])[0].reshape(7, 5)
        axes[row, col].imshow(reconstructed, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_title(f"'{char_labels[i]}'", fontsize=9, fontweight='bold', color='blue')
        axes[row, col].axis('off')
    
    # Agregar etiquetas de sección
    fig.text(0.015, 0.75, 'ORIGINALES', fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center')
    fig.text(0.015, 0.25, 'RECONSTRUIDOS', fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center', color='blue')
    
    plt.suptitle('Reconstrucción Completa del Dataset (32 caracteres)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig('results/all_reconstructions.png', dpi=200, bbox_inches='tight')
    print("Todas las reconstrucciones guardadas en 'all_reconstructions.png'")
    plt.show()

def plot_latent_space(autoencoder, X, original_chars):
    """Visualiza el espacio latente 2D: puntos con etiquetas de texto"""
    # Obtener representaciones latentes
    Z = autoencoder.get_latent_representation(X)
    
    # Caracteres ASCII correspondientes (desde 0x60 hasta 0x7f)
    char_labels = []
    for i in range(len(original_chars)):
        ascii_val = 0x60 + i
        if ascii_val < 0x7f:
            char_labels.append(chr(ascii_val))
        else:
            char_labels.append('DEL')
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calcular márgenes
    z_range_x = Z[:, 0].max() - Z[:, 0].min() if Z[:, 0].max() != Z[:, 0].min() else 1.0
    z_range_y = Z[:, 1].max() - Z[:, 1].min() if Z[:, 1].max() != Z[:, 1].min() else 1.0
    margin = max(z_range_x, z_range_y) * 0.1
    
    # Plotear puntos
    ax.scatter(Z[:, 0], Z[:, 1], s=100, c='blue', alpha=0.6, edgecolors='black', linewidths=1.5, zorder=2)
    
    # Agregar etiquetas de texto encima de cada punto
    for z, label in zip(Z, char_labels):
        ax.text(z[0], z[1], f"'{label}'", 
               ha='center', va='bottom', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5),
               zorder=3)
    
    # Configurar ejes
    ax.set_xlim(Z[:, 0].min() - margin, Z[:, 0].max() + margin)
    ax.set_ylim(Z[:, 1].min() - margin, Z[:, 1].max() + margin)
    ax.set_xlabel('Dimensión Latente 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimensión Latente 2', fontsize=12, fontweight='bold')
    ax.set_title('Espacio Latente 2D - Representación de Caracteres', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('results/latent_space.png', dpi=200, bbox_inches='tight')
    print("Espacio latente guardado en 'latent_space.png'")
    plt.show()

def plot_loss_history(autoencoder):
    """Visualiza la historia de pérdida"""
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.loss_history)
    plt.xlabel('Época')
    plt.ylabel('Error (MSE)')
    plt.title('Historia de Error del Autoencoder')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss_history.png', dpi=150, bbox_inches='tight')
    print("Historia de pérdida guardada en 'loss_history.png'")
    plt.show()

def load_config(config_path='config.json'):
    """Carga la configuración desde un archivo JSON"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def main():
    # Cargar configuración
    print("Cargando configuración...")
    try:
        config = load_config('config.json')
        print("✓ Configuración cargada exitosamente")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Usando configuración por defecto...")
        config = {
            "learning_rate": 0.01,
            "optimizer": "adam",
            "activation": "tanh",
            "encoder_layers": [35, 16, 8, 2],
            "decoder_layers": [2, 8, 16, 35],
            "epochs": 5000,
            "epsilon": 1e-6,
            "batch_size": None,
            "seed": 42
        }
    
    # Mostrar configuración
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL AUTOENCODER")
    print("="*50)
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Optimizador: {config['optimizer']}")
    print(f"Función de Activación: {config['activation']}")
    print(f"Arquitectura Encoder: {config['encoder_layers']}")
    print(f"Arquitectura Decoder: {config['decoder_layers']}")
    print(f"Épocas: {config['epochs']}")
    print(f"Batch Size: {config.get('batch_size', 'Completo')}")
    print(f"Seed: {config['seed']}")
    print("="*50 + "\n")
    
    print("Preparando datos...")
    X = prepare_data()
    print(f"Datos preparados: {X.shape} (32 caracteres, 35 características cada uno)")
    
    # Crear autoencoder con configuración
    print("\nCreando autoencoder...")
    autoencoder = Autoencoder(
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        activation=config['activation'],
        eta=config['learning_rate'],
        optimizer=config['optimizer'],
        batch_size=config.get('batch_size'),
        seed=config['seed']
    )
    
    # Entrenar
    print("\nEntrenando autoencoder...")
    autoencoder.train(
        X, 
        epochs=config['epochs'], 
        epsilon=config.get('epsilon', 1e-6), 
        verbose=True
    )
    
    # Visualizar resultados
    print("\nVisualizando resultados...")
    plot_loss_history(autoencoder)
    plot_reconstructions(autoencoder, X, font_3, n_samples=8)
    plot_all_reconstructions(autoencoder, X, font_3)
    plot_latent_space(autoencoder, X, font_3)
    
    print("\n¡Completado!")

if __name__ == "__main__":
    main()


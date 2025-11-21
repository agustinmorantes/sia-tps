import numpy as np
import matplotlib.pyplot as plt
import json
import os
from seaborn import heatmap
from font import font_3, to_bin_array, monocromatic_cmap
from denoising_autoencoder_simple import DenoisingAutoencoderSimple

def prepare_data():
    data = []
    for caracter in font_3:
        bin_array = to_bin_array(caracter)
        flattened = bin_array.flatten()
        data.append(flattened)
    
    X = np.array(data, dtype=np.float32)
    return X

def plot_reconstructions(autoencoder, X, original_chars, n_samples=8):
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    for i, idx in enumerate(indices):
        original = X[idx].reshape(7, 5)
        
        # Agregar ruido
        X_noisy = autoencoder.add_gaussian_noise(X[idx:idx+1])
        noisy = X_noisy[0].reshape(7, 5)
        
        # Reconstrucción
        reconstructed = autoencoder.denoise(X_noisy)[0].reshape(7, 5)
        
        # Original
        axes[0, i].imshow(original, cmap='binary', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')
        
        # Con ruido
        axes[1, i].imshow(noisy, cmap='binary', vmin=0, vmax=1)
        axes[1, i].set_title(f'Con ruido')
        axes[1, i].axis('off')
        
        # Reconstruido (limpio)
        axes[2, i].imshow(reconstructed, cmap='binary', vmin=0, vmax=1)
        axes[2, i].set_title(f'Limpio')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/denoising_simple_reconstructions.png', dpi=150, bbox_inches='tight')
    print("Reconstrucciones guardadas en 'denoising_simple_reconstructions.png'")
    plt.show()

def plot_all_reconstructions(autoencoder, X, original_chars):
    n_chars = len(X)
    
    char_labels = []
    for i in range(n_chars):
        ascii_val = 0x60 + i
        if ascii_val < 0x7f:
            char_labels.append(chr(ascii_val))
        else:
            char_labels.append('DEL')
    
    n_cols = 8
    n_rows = 4
    
    fig, axes = plt.subplots(3 * n_rows, n_cols, figsize=(20, 15))
    
    # Primera sección: Originales
    for i in range(n_chars):
        row = i // n_cols
        col = i % n_cols
        original = X[i].reshape(7, 5)
        axes[row, col].imshow(original, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_title(f"'{char_labels[i]}'", fontsize=9, fontweight='bold')
        axes[row, col].axis('off')
    
    # Segunda sección: Con ruido
    for i in range(n_chars):
        row = (i // n_cols) + n_rows
        col = i % n_cols
        X_noisy = autoencoder.add_gaussian_noise(X[i:i+1])
        noisy = X_noisy[0].reshape(7, 5)
        axes[row, col].imshow(noisy, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_title(f"'{char_labels[i]}'", fontsize=9, fontweight='bold', color='red')
        axes[row, col].axis('off')
    
    # Tercera sección: Limpios
    for i in range(n_chars):
        row = (i // n_cols) + 2 * n_rows
        col = i % n_cols
        X_noisy = autoencoder.add_gaussian_noise(X[i:i+1])
        reconstructed = autoencoder.denoise(X_noisy)[0].reshape(7, 5)
        axes[row, col].imshow(reconstructed, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_title(f"'{char_labels[i]}'", fontsize=9, fontweight='bold', color='blue')
        axes[row, col].axis('off')
    
    fig.text(0.015, 0.83, 'ORIGINALES', fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center')
    fig.text(0.015, 0.5, 'CON RUIDO', fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center', color='red')
    fig.text(0.015, 0.17, 'LIMPIO', fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center', color='blue')
    
    plt.suptitle('Denoising Autoencoder - Comparación Completa', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig('results/denoising_simple_all_reconstructions.png', dpi=200, bbox_inches='tight')
    print("Todas las reconstrucciones guardadas en 'denoising_simple_all_reconstructions.png'")
    plt.show()

def plot_loss_history(autoencoder):
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.loss_history)
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.title('Historia de Pérdida del Denoising Autoencoder (MLP Único)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/denoising_simple_loss_history.png', dpi=150, bbox_inches='tight')
    print("Historia de pérdida guardada en 'denoising_simple_loss_history.png'")
    plt.show()

def load_config(config_path='config.json'):
    """Carga la configuración desde un archivo JSON"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def main():
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
            "seed": 42,
            "noise_std": 0.1
        }
    
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL DENOISING AUTOENCODER")
    print("="*50)
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Optimizador: {config['optimizer']}")
    print(f"Función de Activación: {config['activation']}")
    print(f"Arquitectura Encoder: {config['encoder_layers']}")
    print(f"Arquitectura Decoder: {config['decoder_layers']}")
    print(f"Arquitectura Completa: {config['encoder_layers'] + config['decoder_layers'][1:]}")
    print(f"Épocas: {config['epochs']}")
    print(f"Batch Size: {config.get('batch_size', 'Completo')}")
    print(f"Seed: {config['seed']}")
    print(f"Noise Std (Ruido Gaussiano): {config.get('noise_std', 0.1)}")
    print("="*50 + "\n")
    
    print("Preparando datos...")
    X = prepare_data()
    print(f"Datos preparados: {X.shape} (32 caracteres, 35 características cada uno)")
    
    print("\nCreando Denoising Autoencoder (MLP único)...")
    autoencoder = DenoisingAutoencoderSimple(
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        activation=config['activation'],
        eta=config['learning_rate'],
        optimizer=config['optimizer'],
        batch_size=config.get('batch_size'),
        seed=config['seed'],
        noise_std=config.get('noise_std', 0.1)
    )
    
    print("\nEntrenando Denoising Autoencoder...")
    print("(Entrada: X con ruido gaussiano, Target: X sin ruido)")
    autoencoder.train(
        X, 
        epochs=config['epochs'], 
        epsilon=config.get('epsilon', 1e-6), 
        verbose=True
    )
    
    print("\nVisualizando resultados...")
    plot_loss_history(autoencoder)
    plot_reconstructions(autoencoder, X, font_3, n_samples=8)
    plot_all_reconstructions(autoencoder, X, font_3)

if __name__ == "__main__":
    main()


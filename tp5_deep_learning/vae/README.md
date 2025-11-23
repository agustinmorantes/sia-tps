# Variational Autoencoder (VAE)

Implementación de un Variational Autoencoder que hereda de `AutoencoderSimple`.

## Estructura de archivos

```
vae/
├── config.json          # Configuración del VAE
├── vae.py              # Clase VariationalAutoencoder
├── main_vae.py         # Script de entrenamiento y visualización
├── results/            # Directorio de salida (se crea automáticamente)
└── README.md           # Este archivo
```

## Configuración (`config.json`)

- `activation`: Función de activación ('tanh' o 'sigmoid')
- `optimizer`: Optimizador para el decoder ('sgd', 'momentum', 'adam')
- `seed`: Semilla para reproducibilidad
- `epsilon`: Umbral de convergencia
- `vae_learning_rate`: Learning rate del VAE
- `vae_h1`, `vae_h2`: Tamaños de capas ocultas del encoder
- `vae_latent_dim`: Dimensión del espacio latente
- `vae_beta`: Peso del término KL en la pérdida (β-VAE)
- `vae_epochs`: Número máximo de épocas
- `vae_batch_size`: Tamaño del batch (null = full batch)
- `icons_dir`: Ruta relativa a la carpeta de iconos
- `results_dir`: Directorio donde se guardan las visualizaciones

## Uso

```bash
# Ejecutar desde la carpeta vae/
python main_vae.py
```

El script:
1. Carga los iconos desde `../resources/icons/`
2. Entrena el VAE con la configuración especificada
3. Genera visualizaciones en `results/`:
   - `vae_training_history.png`: Evolución de la pérdida
   - `vae_reconstructions.png`: Originales vs reconstrucciones
   - `vae_latent_space.png`: Distribución en el espacio latente 2D
   - `vae_generated_samples.png`: Nuevas muestras generadas
   - `vae_interpolation_X_to_Y.png`: Interpolaciones entre iconos

## Arquitectura del VAE

El VAE consta de:

1. **Encoder**: Produce dos salidas (μ y log σ²) para cada entrada
2. **Reparametrización**: z = μ + ε · σ, donde ε ~ N(0, I)
3. **Decoder**: Reconstruye la entrada desde z

La función de pérdida combina:
- Pérdida de reconstrucción (MSE)
- Divergencia KL entre q(z|x) y N(0, I)

## Métodos disponibles

- `train(X, epochs, epsilon, verbose, X_val)`: Entrena el VAE
- `reconstruct(X)`: Reconstruye las entradas
- `generate(n_samples)`: Genera nuevas muestras desde N(0, I)
- `interpolate(X1, X2, n_steps)`: Interpola entre dos muestras
- `sample_from_input(X, n_samples)`: Genera variaciones de una entrada
- `get_latent_representation(X)`: Obtiene μ para las entradas

# TP5 - Deep Learning: Autoencoders

Este proyecto implementa **Autoencoders** y **Denoising Autoencoders** para la compresión y reconstrucción de imágenes binarias de caracteres ASCII.

## 📁 Estructura del Proyecto

```
tp5_deep_learning/
├── font.py                          # Dataset de caracteres binarios
├── multi_layer_perceptron.py        # Implementación del MLP base
├── autoencoder_simple.py            # Autoencoder
├── denoising_autoencoder_simple.py  # Denoising Autoencoder
├── main_simple.py                   # Script principal para Autoencoder
├── main_denoising_simple.py         # Script principal para Denoising Autoencoder
├── config.json                      # Configuración de hiperparámetros
├── results/                          # Gráficos y resultados generados
└── vae/                             # Variational Autoencoder (implementación adicional)
```

## ⚙️ Configuración

El archivo `config.json` permite configurar los hiperparámetros:

```json
{
  "learning_rate": 0.005,
  "optimizer": "adam",
  "activation": "tanh",
  "encoder_layers": [35, 30, 20, 10, 2],
  "decoder_layers": [2, 10, 20, 30, 35],
  "epochs": 10000,
  "epsilon": 1e-6,
  "batch_size": null,
  "seed": 42,
  "noise_std": 1.0
}
```

### Parámetros explicados

- **`learning_rate`**: Tasa de aprendizaje 
- **`optimizer`**: Optimizador (`"sgd"`, `"adam"`)
- **`activation`**: Función de activación (`"tanh"` o `"sigmoid"`)
- **`encoder_layers`**: Arquitectura del encoder `[input, hidden1, ..., latent_dim]`
- **`decoder_layers`**: Arquitectura del decoder `[latent_dim, hidden1, ..., output]`
- **`epochs`**: Número de épocas de entrenamiento
- **`batch_size`**: Tamaño del batch (`null` = batch completo)
- **`seed`**: Semilla para reproducibilidad
- **`noise_std`**: Desviación estándar del ruido gaussiano (solo para Denoising Autoencoder)

## 🎯 Uso

### Autoencoder Básico

```bash
python main_simple.py
```

Este script:
1. Carga la configuración desde `config.json`
2. Prepara los datos del dataset
3. Entrena el autoencoder
4. Genera visualizaciones:
   - Historial de pérdida
   - Reconstrucciones (muestra y todas)
   - Espacio latente 2D
   - Interpolaciones entre caracteres

### Denoising Autoencoder

```bash
python main_denoising_simple.py
```

Similar al autoencoder básico, pero:
- Agrega ruido gaussiano a los datos de entrada durante el entrenamiento
- Usa los datos originales (sin ruido) como target
- Aprende a reconstruir caracteres corruptos

## 📊 Resultados Generados

Los resultados se guardan en `results/`:

### Autoencoder Variacional

**Aclaración: para obtener más información sobre el Autoencoder Variacional dirigirse al README dentro de la carpeta llamada `vae/`**

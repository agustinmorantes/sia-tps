import numpy as np
from autoencoder_simple import AutoencoderSimple

class DenoisingAutoencoderSimple(AutoencoderSimple):
    
    def __init__(self, encoder_layers=[35, 16, 8, 2], decoder_layers=[2, 8, 16, 35],
                 activation="tanh", eta=0.05, alpha=0.9, optimizer='adam', 
                 batch_size=None, seed=123, noise_std=0.1):
        """
        Args:
            encoder_layers: Arquitectura del encoder [input, hidden1, ..., latent_dim]
            decoder_layers: Arquitectura del decoder [latent_dim, hidden1, ..., output]
            activation: Función de activación ("tanh" o "sigmoid")
            eta: Learning rate
            alpha: Momentum (si se usa)
            optimizer: 'sgd', 'momentum', o 'adam'
            batch_size: Tamaño del batch (None = batch completo)
            seed: Semilla para reproducibilidad
            noise_std: Desviación estándar del ruido gaussiano (default: 0.1)
        """
        super().__init__(encoder_layers, decoder_layers, activation, eta, alpha, 
                        optimizer, batch_size, seed)
        self.noise_std = noise_std
    
    def add_gaussian_noise(self, X):
        noise = np.random.normal(0, self.noise_std, size=X.shape)
        X_noisy = X + noise
        
        # Para datos binarios, recortar entre 0 y 1
        X_noisy = np.clip(X_noisy, 0, 1)
        
        return X_noisy
    
    def train(self, X, epochs=10000, epsilon=1e-5, verbose=True):
        n_samples = X.shape[0]
        effective_batch_size = n_samples if self.mlp.batch_size is None else self.mlp.batch_size
        
        for epoch in range(epochs):
            # Shuffle de los datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, n_samples)
                X_batch_clean = X_shuffled[start_idx:end_idx]
                
                # Agregar ruido gaussiano a la entrada
                X_batch_noisy = self.add_gaussian_noise(X_batch_clean)
                
                # Forward pass completo: X_noisy -> MLP -> X_reconstructed
                activations = self.mlp.forward(X_batch_noisy)
                X_reconstructed = activations[-1]
                
                # Calcular pérdida: comparar con X original (sin ruido)
                batch_loss = self.compute_loss(X_batch_clean, X_reconstructed)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass: el MLP maneja todo automáticamente
                # Target es X_batch_clean (sin ruido)
                deltas = self.mlp.backward(activations, X_batch_clean)
                self.mlp.update_weights(activations, deltas)
            
            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs}, Pérdida: {epoch_loss:.6f}")
            
            if epoch_loss < epsilon:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch + 1}")
                self.converged_epoch = epoch + 1
                break
        else:
            self.converged_epoch = epochs
            if verbose:
                print(f"Entrenamiento completado después de {epochs} épocas")
    
    def denoise(self, X_noisy):
        return self.reconstruct(X_noisy)


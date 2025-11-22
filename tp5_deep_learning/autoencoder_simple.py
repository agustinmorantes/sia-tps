import numpy as np
from multi_layer_perceptron import MultiLayerPerceptron

class AutoencoderSimple:
    
    def __init__(self, encoder_layers=[35, 16, 8, 2], decoder_layers=[2, 8, 16, 35],
                 activation="tanh", eta=0.05, alpha=0.9, optimizer='adam', 
                 batch_size=None, seed=123):
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
        """
        # encoder_layers = [35, 16, 8, 2]
        # decoder_layers = [2, 8, 16, 35]
        # Arquitectura completa: [35, 16, 8, 2, 8, 16, 35]
        full_layers = encoder_layers + decoder_layers[1:]  # Evitar duplicar el espacio latente
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.latent_dim = encoder_layers[-1]
        # Índice de activación donde está Z: después de todas las capas del encoder
        # activations[0] = entrada, activations[1] = después capa 0, ..., 
        # activations[len(encoder_layers)-1] = después última capa encoder = Z
        self.latent_activation_idx = len(encoder_layers) - 1  # Índice de activación donde está Z
        self.latent_layer_idx = len(encoder_layers) - 1  # Índice de la última capa de pesos del encoder
        
        # Crear un único MLP con arquitectura simétrica
        self.mlp = MultiLayerPerceptron(
            layer_sizes=full_layers,
            activation=activation,
            eta=eta,
            alpha=alpha,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=seed
        )
        
        self.loss_history = []
    
    def encode(self, X):
        # Forward hasta la capa latente
        activations = self.mlp.forward(X)
        return activations[self.latent_activation_idx]
    
    def decode(self, Z):
        # Hacer forward manual desde Z (capa latente) hacia la salida
        current = Z
        start_layer = self.latent_layer_idx
        for i in range(start_layer, self.mlp.n_layers):
            z = np.dot(current, self.mlp.weights[i]) + self.mlp.biases[i]
            current = self.mlp.activation.fn(z)
        return current
    
    def forward(self, X):
        """Paso forward completo: X -> encoder -> Z -> decoder -> X_reconstructed"""
        activations = self.mlp.forward(X)
        Z = activations[self.latent_activation_idx]
        X_reconstructed = activations[-1]
        return X_reconstructed, Z
    
    def compute_loss(self, X, X_reconstructed):
        return np.mean((X - X_reconstructed) ** 2)
    
    def train(self, X, epochs=10000, epsilon=1e-5, verbose=True): #X es la matriz de 32 x 35 
        n_samples = X.shape[0]
        effective_batch_size = n_samples if self.mlp.batch_size is None else self.mlp.batch_size
        
        for epoch in range(epochs):
            # Shuffle de los datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices] #Mezclo las filas 
            
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, effective_batch_size): #Con batch_size=None, procesa todos los 32 caracteres juntos
                end_idx = min(start_idx + effective_batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                
                # Forward pass completo
                activations = self.mlp.forward(X_batch) # Le paso los 32 caracteres por batch=null
                X_reconstructed = activations[-1]
                
                # Calcular pérdida
                batch_loss = self.compute_loss(X_batch, X_reconstructed)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass: el MLP maneja todo automáticamente
                # Target es X_batch
                deltas = self.mlp.backward(activations, X_batch)
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
    
    def reconstruct(self, X):
        return self.forward(X)[0]
    
    def get_latent_representation(self, X):
        return self.encode(X)


import numpy as np
from multi_layer_perceptron import MultiLayerPerceptron

class Autoencoder:
    
    def __init__(self, encoder_layers=[35, 16, 8, 2], decoder_layers=[2, 8, 16, 35],
                 activation="tanh", eta=0.05, alpha=0.9, optimizer='adam', 
                 batch_size=None, seed=123):
                 
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.latent_dim = encoder_layers[-1]
        
        # Crear encoder y decoder
        self.encoder = MultiLayerPerceptron(
            layer_sizes=encoder_layers,
            activation=activation,
            eta=eta,
            alpha=alpha,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=seed
        )
        
        # Para el decoder, usamos sigmoid en la última capa si es necesario
        # Por ahora usamos la misma activación
        self.decoder = MultiLayerPerceptron(
            layer_sizes=decoder_layers,
            activation=activation,
            eta=eta,
            alpha=alpha,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=seed+1 if seed is not None else None
        )
        
        self.loss_history = []
    
    def encode(self, X):
        """Codifica la entrada X al espacio latente"""
        return self.encoder.predict(X)
    
    def decode(self, Z):
        """Decodifica el espacio latente Z a la reconstrucción"""
        return self.decoder.predict(Z)
    
    def forward(self, X):
        """Paso forward completo: X -> encoder -> Z -> decoder -> X_reconstructed"""
        Z = self.encode(X)
        X_reconstructed = self.decode(Z)
        return X_reconstructed, Z
    
    def compute_loss(self, X, X_reconstructed):
        """Calcula la pérdida MSE (Mean Squared Error) entre X y X_reconstructed"""
        return np.mean((X - X_reconstructed) ** 2)
    
    def train(self, X, epochs=10000, epsilon=1e-5, verbose=True):
        """
        Entrena el autoencoder usando backpropagation end-to-end.
        En un autoencoder, la salida esperada es la misma que la entrada (auto-supervisado).
        
        Args:
            X: Datos de entrada (n_samples, n_features)
            epochs: Número máximo de épocas
            epsilon: Umbral de convergencia
            verbose: Si True, imprime el progreso
        """
        n_samples = X.shape[0]
        effective_batch_size = n_samples if self.encoder.batch_size is None else self.encoder.batch_size
        
        for epoch in range(epochs):
            # Shuffle de los datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                
                # Forward pass completo
                encoder_activations = self.encoder.forward(X_batch)
                Z = encoder_activations[-1]
                decoder_activations = self.decoder.forward(Z)
                X_reconstructed = decoder_activations[-1]
                
                # Calcular pérdida
                batch_loss = self.compute_loss(X_batch, X_reconstructed)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass completo end-to-end
                # 1. Entrenar decoder: propagar error hacia atrás
                decoder_deltas = self.decoder.backward(decoder_activations, X_batch)
                self.decoder.update_weights(decoder_activations, decoder_deltas)
                
                # 3. Propagar error hacia el encoder a través del decoder
                # decoder_deltas[0] es el delta de la primera capa oculta del decoder
                # Este delta ya tiene la derivada de activación aplicada
                if len(decoder_deltas) > 0 and len(self.decoder.weights) > 0:
                    # Calcular el gradiente en Z (salida del encoder, entrada del decoder)
                    # decoder_deltas[0] es el error en la primera capa oculta del decoder
                    # Necesitamos propagarlo hacia Z multiplicando por los pesos
                    # grad_Z = decoder_deltas[0] @ decoder.weights[0].T
                    # Este gradiente está DESPUÉS de la activación del encoder (Z ya pasó por tanh/sigmoid)
                    grad_Z = np.dot(decoder_deltas[0], self.decoder.weights[0].T)
                    
                    # 4. Calcular deltas del encoder manualmente (backpropagation correcto)
                    # grad_Z es el gradiente en Z (después de la activación del encoder)
                    # Para obtener el delta de la última capa del encoder, necesitamos:
                    # delta = grad_Z * activation.derivative(Z)
                    # Esto nos da el gradiente antes de la activación de la última capa
                    
                    encoder_deltas = []
                    
                    # Delta de la última capa del encoder
                    # Z es la salida activada, así que aplicamos la derivada
                    delta_last = grad_Z * self.encoder.activation.derivative(Z)
                    encoder_deltas.insert(0, delta_last)
                    
                    # Propagar hacia atrás a través de las capas restantes del encoder
                    current_delta = delta_last
                    for i in reversed(range(self.encoder.n_layers - 1)):
                        # Propagar el delta hacia la capa anterior
                        # delta[i] = delta[i+1] @ weights[i+1].T * activation.derivative(activations[i+1])
                        delta = np.dot(current_delta, self.encoder.weights[i+1].T) * \
                               self.encoder.activation.derivative(encoder_activations[i+1])
                        encoder_deltas.insert(0, delta)
                        current_delta = delta
                    
                    # 5. Actualizar pesos del encoder usando los deltas calculados
                    self.encoder.t += 1
                    for i in range(self.encoder.n_layers):
                        grad_w = np.dot(encoder_activations[i].T, encoder_deltas[i])
                        grad_b = np.sum(encoder_deltas[i], axis=0, keepdims=True)
                        
                        if self.encoder.optimizer == 'sgd':
                            self.encoder.weights[i] -= self.encoder.eta * grad_w
                            self.encoder.biases[i] -= self.encoder.eta * grad_b
                        
                        elif self.encoder.optimizer == 'momentum':
                            deltaW = self.encoder.eta * grad_w + self.encoder.alpha * self.encoder.deltaW_prev[i]
                            deltaB = self.encoder.eta * grad_b + self.encoder.alpha * self.encoder.deltaB_prev[i]
                            self.encoder.weights[i] -= deltaW
                            self.encoder.biases[i] -= deltaB
                            self.encoder.deltaW_prev[i] = deltaW
                            self.encoder.deltaB_prev[i] = deltaB
                        
                        elif self.encoder.optimizer == 'adam':
                            beta1, beta2, eps = 0.9, 0.999, 1e-8
                            self.encoder.m_w[i] = beta1 * self.encoder.m_w[i] + (1 - beta1) * grad_w
                            self.encoder.v_w[i] = beta2 * self.encoder.v_w[i] + (1 - beta2) * (grad_w**2)
                            m_hat_w = self.encoder.m_w[i] / (1 - beta1**self.encoder.t)
                            v_hat_w = self.encoder.v_w[i] / (1 - beta2**self.encoder.t)
                            self.encoder.weights[i] -= self.encoder.eta * m_hat_w / (np.sqrt(v_hat_w) + eps)
                            
                            self.encoder.m_b[i] = beta1 * self.encoder.m_b[i] + (1 - beta1) * grad_b
                            self.encoder.v_b[i] = beta2 * self.encoder.v_b[i] + (1 - beta2) * (grad_b**2)
                            m_hat_b = self.encoder.m_b[i] / (1 - beta1**self.encoder.t)
                            v_hat_b = self.encoder.v_b[i] / (1 - beta2**self.encoder.t)
                            self.encoder.biases[i] -= self.encoder.eta * m_hat_b / (np.sqrt(v_hat_b) + eps)
            
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
        """Reconstruye las entradas X"""
        return self.forward(X)[0]
    
    def get_latent_representation(self, X):
        """Obtiene la representación en el espacio latente de X"""
        return self.encode(X)


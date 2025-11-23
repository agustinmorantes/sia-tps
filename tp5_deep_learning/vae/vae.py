import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoencoder_simple import AutoencoderSimple
from multi_layer_perceptron import MultiLayerPerceptron


class VariationalAutoencoder(AutoencoderSimple):
    """
    Variational Autoencoder (VAE) que hereda de AutoencoderSimple.

    Diferencias con el autoencoder determinista:
    - El encoder ya no produce Z directo, sino dos salidas: mu(x) y logvar(x).
    - La variable latente se obtiene por el truco de reparametrización:
          z = mu + eps * exp(0.5 * logvar)
    - La función de costo incluye:
          L = L_reconstruccion + KL(q(z|x) || N(0, I))

      donde:
          L_reconstruccion = MSE(X, X_reconstruido)
          KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(self,
                 encoder_layers=[35, 16, 8, 2],
                 decoder_layers=[2, 8, 16, 35],
                 activation="tanh",
                 eta=0.05,
                 alpha=0.9,
                 optimizer='sgd',
                 batch_size=None,
                 seed=123,
                 kl_weight=1.0):
        """
        Args:
            encoder_layers: [input, h1, ..., h_L, latent_dim]
            decoder_layers: [latent_dim, h1, ..., output]
            activation: 'tanh' o 'sigmoid'
            eta: learning rate
            alpha: momentum (no se usa en este VAE, pero se pasa al decoder MLP)
            optimizer: 'sgd' | 'momentum' | 'adam' (solo decoder MLP)
            batch_size: no usado aquí (se entrena full batch)
            seed: semilla para inicializar pesos
            kl_weight: λ para ponderar el término KL en la loss total
        """
        # Llamamos al init del AutoencoderSimple principalmente para reutilizar
        # encoder_layers, decoder_layers, latent_dim y parámetros de training.
        super().__init__(
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            eta=eta,
            alpha=alpha,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=seed
        )

        self.kl_weight = kl_weight

        # Usaremos la activación del MLP de la clase base (tanh o sigmoid)
        self.activation = self.mlp.activation  # objeto ActivationFunction (fn, derivative)

        # -------------------------
        #  ENCODER PROPIO DEL VAE
        # -------------------------
        # encoder_layers = [input, h1, ..., h_L, latent_dim]
        # Usamos hasta h_L como tronco determinista; latent_dim es el tamaño del espacio latente.
        self.encoder_hidden_sizes = self.encoder_layers[:-1]   # [input, h1, ..., h_L]
        self.latent_dim = self.encoder_layers[-1]

        rng = np.random.RandomState(seed)

        self.encoder_weights = []
        self.encoder_biases = []

        for i in range(len(self.encoder_hidden_sizes) - 1):
            in_dim = self.encoder_hidden_sizes[i]
            out_dim = self.encoder_hidden_sizes[i + 1]
            W = rng.uniform(-0.5, 0.5, (in_dim, out_dim))
            b = rng.uniform(-0.5, 0.5, (1, out_dim))
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)

        # Cabezas lineales para mu(x) y logvar(x)
        last_hidden_dim = self.encoder_hidden_sizes[-1]

        self.W_mu = rng.uniform(-0.5, 0.5, (last_hidden_dim, self.latent_dim))
        self.b_mu = np.zeros((1, self.latent_dim))

        self.W_logvar = rng.uniform(-0.5, 0.5, (last_hidden_dim, self.latent_dim))
        self.b_logvar = np.zeros((1, self.latent_dim))

        # Estados para optimizer simple (SGD) – si querés, acá podrías extender a momentum/adam
        self.encoder_eta = eta
        self.head_eta = eta

        # -------------------------
        #  DECODER: MLP estándar
        # -------------------------
        # Usamos un MLP independiente solo para el decoder.
        self.decoder = MultiLayerPerceptron(
            layer_sizes=self.decoder_layers,  # [latent_dim, ..., output_dim]
            activation=activation,
            eta=eta,
            alpha=alpha,
            optimizer=optimizer,
            batch_size=None,   # full batch
            seed=seed
        )

        self.loss_history = []

    # =====================================================
    #                   FORWARD PASS
    # =====================================================
    def encoder_forward(self, X):
        """
        Forward determinista del encoder hasta el último hidden h(x).
        Devuelve:
            activations: [a0=X, a1, ..., a_L=h]
        """
        activations = [X]
        a = X
        for W, b in zip(self.encoder_weights, self.encoder_biases):
            z = np.dot(a, W) + b
            a = self.activation.fn(z)
            activations.append(a)
        return activations  # última activación es h(x)

    def encode(self, X):
        """
        Encoder completo: X -> h(X) -> mu(X), logvar(X)
        """
        activations = self.encoder_forward(X)
        h = activations[-1]

        mu = np.dot(h, self.W_mu) + self.b_mu
        logvar = np.dot(h, self.W_logvar) + self.b_logvar

        return mu, logvar, activations

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Trick de reparametrización:
            z = mu + eps * exp(0.5 * logvar)
        donde eps ~ N(0, I).
        """
        eps = np.random.normal(0.0, 1.0, size=mu.shape)
        std = np.exp(0.5 * logvar)
        z = mu + eps * std
        return z, eps, std

    def decode(self, Z):
        """
        Decoder estándar: usa el MLP del decoder.
        """
        activations = self.decoder.forward(Z)
        X_recon = activations[-1]
        return X_recon, activations

    def forward(self, X):
        """
        Forward completo del VAE: X -> (mu, logvar) -> z -> decoder -> X_recon
        Devuelve:
            X_recon, mu, logvar, z, activations_encoder, activations_decoder, eps, std
        """
        mu, logvar, enc_activations = self.encode(X)
        z, eps, std = self.reparameterize(mu, logvar)
        X_recon, dec_activations = self.decode(z)

        return X_recon, mu, logvar, z, enc_activations, dec_activations, eps, std

    # =====================================================
    #                   LOSS DEL VAE
    # =====================================================
    def compute_losses(self, X, X_recon, mu, logvar):
        """
        Calcula:
            - recon_loss (MSE)
            - KL divergence
            - total_loss = recon + λ * KL
        """
        # MSE de reconstrucción
        recon_loss = np.mean((X - X_recon) ** 2)

        # KL(q(z|x) || N(0, I)):
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_sample = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)
        kl_loss = np.mean(kl_per_sample)

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    # =====================================================
    #                   BACKPROP
    # =====================================================
    def train(self, X, epochs=5000, epsilon=1e-5, verbose=True, X_val=None):
        """
        Entrenamiento del VAE usando full batch (todos los samples a la vez).
        - Decoder: se actualiza con gradiente de reconstrucción.
        - Encoder: se actualiza con gradiente de reconstrucción + gradiente del KL.
        
        Args:
            X: Datos de entrenamiento
            epochs: Número máximo de épocas
            epsilon: Umbral de convergencia
            verbose: Si mostrar mensajes de progreso
            X_val: Datos de validación opcionales
        """
        n_samples = X.shape[0]
        
        # Historial de pérdidas de validación
        self.val_loss_history = [] if X_val is not None else None

        for epoch in range(epochs):
            # -------------------------
            # FORWARD PASS
            # -------------------------
            X_recon, mu, logvar, z, enc_acts, dec_acts, eps, std = self.forward(X)

            # -------------------------
            # LOSS
            # -------------------------
            total_loss, recon_loss, kl_loss = self.compute_losses(X, X_recon, mu, logvar)
            self.loss_history.append(total_loss)
            
            # Validación si se proporcionan datos
            if X_val is not None:
                X_val_recon, mu_val, logvar_val, _, _, _, _, _ = self.forward(X_val)
                val_total_loss, val_recon_loss, val_kl_loss = self.compute_losses(
                    X_val, X_val_recon, mu_val, logvar_val
                )
                self.val_loss_history.append(val_total_loss)

            # -------------------------
            # BACKPROP DECODER (reconstrucción)
            # -------------------------
            # Usamos el backward del MLP del decoder, tomando X como "target".
            dec_deltas = self.decoder.backward(dec_acts, X)
            self.decoder.update_weights(dec_acts, dec_deltas)

            # Necesitamos ∂L_recon/∂z para el encoder.
            # Primer delta es para la primera capa del decoder (sobre preactivación),
            # gradiente respecto de z (entrada) = delta_0 * W0^T
            dLrecon_dz = np.dot(dec_deltas[0], self.decoder.weights[0].T)

            # -------------------------
            # GRADIENTES MU & LOGVAR
            # -------------------------
            # Derivadas del término KL (por muestra):
            # d(KL)/d(mu) = mu
            # d(KL)/d(logvar) = 0.5 * (exp(logvar) - 1)
            dKL_dmu = mu
            dKL_dlogvar = 0.5 * (np.exp(logvar) - 1)

            # Contribución de reconstrucción vía reparametrización:
            # z = mu + eps * std,   std = exp(0.5 * logvar)
            # dz/dmu = 1
            # dz/dlogvar = eps * 0.5 * std
            dz_dmu = 1.0
            dz_dlogvar = eps * 0.5 * std

            # Gradiente total wrt mu y logvar:
            dL_dmu = dLrecon_dz * dz_dmu + self.kl_weight * dKL_dmu
            dL_dlogvar = dLrecon_dz * dz_dlogvar + self.kl_weight * dKL_dlogvar

            # -------------------------
            # BACKPROP CABEZAS MU & LOGVAR
            # -------------------------
            h = enc_acts[-1]  # última activación del encoder

            grad_W_mu = np.dot(h.T, dL_dmu) / n_samples
            grad_b_mu = np.sum(dL_dmu, axis=0, keepdims=True) / n_samples

            grad_W_logvar = np.dot(h.T, dL_dlogvar) / n_samples
            grad_b_logvar = np.sum(dL_dlogvar, axis=0, keepdims=True) / n_samples

            self.W_mu -= self.head_eta * grad_W_mu
            self.b_mu -= self.head_eta * grad_b_mu

            self.W_logvar -= self.head_eta * grad_W_logvar
            self.b_logvar -= self.head_eta * grad_b_logvar

            # Gradiente que baja a h desde ambas cabezas:
            dL_dh_from_mu = np.dot(dL_dmu, self.W_mu.T)
            dL_dh_from_logvar = np.dot(dL_dlogvar, self.W_logvar.T)
            dL_dh = dL_dh_from_mu + dL_dh_from_logvar

            # -------------------------
            # BACKPROP ENCODER (hidden layers)
            # -------------------------
            dA_next = dL_dh  # gradiente respecto de activación de la última capa oculta

            # encoder_weights[i]: de capa i -> i+1
            # enc_acts[i] = activación de capa i (i=0 es X)
            num_layers_enc = len(self.encoder_weights)

            for i in reversed(range(num_layers_enc)):
                a_out = enc_acts[i + 1]  # output de la capa i (después de activación)
                a_in = enc_acts[i]       # input de la capa i

                # delta = dL/dz_i = dL/d(a_out) * f'(a_out)
                delta = dA_next * self.activation.derivative(a_out)

                grad_W = np.dot(a_in.T, delta) / n_samples
                grad_b = np.sum(delta, axis=0, keepdims=True) / n_samples

                # Actualizamos pesos del encoder (SGD)
                self.encoder_weights[i] -= self.encoder_eta * grad_W
                self.encoder_biases[i] -= self.encoder_eta * grad_b

                # gradiente para la capa anterior: dL/d(a_in)
                dA_next = np.dot(delta, self.encoder_weights[i].T)

            # -------------------------
            # Logs
            # -------------------------
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f"Época {epoch+1}/{epochs} | "
                          f"Train Loss: {total_loss:.6f} (Recon: {recon_loss:.6f}, KL: {kl_loss:.6f}) | "
                          f"Val Loss: {val_total_loss:.6f} (Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f})")
                else:
                    print(f"Época {epoch+1}/{epochs} | "
                          f"Loss total: {total_loss:.6f} | "
                          f"Recon: {recon_loss:.6f} | KL: {kl_loss:.6f}")

            if total_loss < epsilon:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch+1}")
                self.converged_epoch = epoch + 1
                break
        else:
            self.converged_epoch = epochs
            if verbose:
                print(f"Entrenamiento completado después de {epochs} épocas")

    # =====================================================
    #      MÉTODOS ÚTILES COMPATIBLES CON AUTOENCODER
    # =====================================================
    def reconstruct(self, X):
        X_recon, _, _, _, _, _, _, _ = self.forward(X)
        return X_recon

    def get_latent_representation(self, X):
        mu, logvar, _ = self.encode(X)
        # Podés devolver solo mu (representación "media") o samplear:
        return mu

    # =====================================================
    #      MÉTODOS GENERATIVOS
    # =====================================================
    def generate(self, n_samples=1):
        """
        Genera nuevas muestras desde la distribución previa N(0, I).
        Útil para crear datos sintéticos del mismo tipo que los de entrenamiento.
        
        Args:
            n_samples: Cantidad de muestras a generar
            
        Returns:
            X_generated: Array de forma (n_samples, output_dim) con las muestras generadas
        """
        # Samplear desde la distribución previa estándar
        z = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Pasar por el decoder
        X_generated, _ = self.decode(z)
        
        return X_generated
    
    def interpolate(self, X1, X2, n_steps=10):
        """
        Interpola entre dos muestras en el espacio latente.
        Útil para visualizar transiciones suaves entre datos.
        
        Args:
            X1: Primera muestra de entrada (1, input_dim) o (input_dim,)
            X2: Segunda muestra de entrada (1, input_dim) o (input_dim,)
            n_steps: Número de pasos de interpolación
            
        Returns:
            X_interpolated: Array de forma (n_steps, output_dim) con las interpolaciones
        """
        # Asegurar que X1 y X2 tengan forma (1, input_dim)
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        # Obtener representaciones latentes (usando mu, sin samplear)
        mu1, _, _ = self.encode(X1)
        mu2, _, _ = self.encode(X2)
        
        # Interpolar linealmente en el espacio latente
        alphas = np.linspace(0, 1, n_steps)
        z_interpolated = np.array([alpha * mu2 + (1 - alpha) * mu1 for alpha in alphas])
        z_interpolated = z_interpolated.reshape(n_steps, self.latent_dim)
        
        # Decodificar las interpolaciones
        X_interpolated, _ = self.decode(z_interpolated)
        
        return X_interpolated
    
    def sample_from_input(self, X, n_samples=5):
        """
        Genera múltiples variaciones de una entrada dada, sampleando desde q(z|x).
        Útil para visualizar la variabilidad del VAE.
        
        Args:
            X: Muestra de entrada (1, input_dim) o (input_dim,)
            n_samples: Cantidad de variaciones a generar
            
        Returns:
            X_variations: Array de forma (n_samples, output_dim) con las variaciones
        """
        # Asegurar que X tenga forma (1, input_dim)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Obtener mu y logvar
        mu, logvar, _ = self.encode(X)
        
        # Generar múltiples samples
        X_variations = []
        for _ in range(n_samples):
            z, _, _ = self.reparameterize(mu, logvar)
            X_recon, _ = self.decode(z)
            X_variations.append(X_recon)
        
        return np.vstack(X_variations)

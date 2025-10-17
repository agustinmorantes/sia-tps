import numpy as np 

class KohonenNetwork:
    def __init__(self, k, learning_rate=1, columns=8, neighborhood_ratio=3, data=None):
        self.k = k
        self.learning_rate = learning_rate
        self.collumns = columns
        self.neighborhood_ratio = neighborhood_ratio
        self.iteration = 0  # Para el decaimiento de learning rate
        
        # Inicializar pesos con muestras aleatorias de los datos
        if data is not None:
            self.map = self._initialize_weights_from_data(data)
        else:
            # Fallback a inicialización aleatoria si no hay datos
            self.map = np.random.rand(self.k, self.k, self.collumns)
    
    def _initialize_weights_from_data(self, data):
        n_samples, n_features = data.shape
        weights = np.zeros((self.k, self.k, n_features)) # Crear matriz de pesos
        for i in range(self.k):    # Para cada neurona, seleccionar una muestra aleatoria de los datos
            for j in range(self.k):
                random_idx = np.random.randint(0, n_samples)   # Seleccionar índice aleatorio de los datos
                weights[i, j] = data[random_idx].copy()
        
        return weights

    def winner_neuron(self, input_vec):
        min_dist = float('inf') # Para que cualquier valor ya lo reemplace en la primera instancia
        winner_pos = (0, 0)
        for i in range(self.k):
            for j in range(self.k):
                weights = self.map[i, j]  
                dist = np.linalg.norm(input_vec - weights) # Distancia euclidiana por ahora 
                if dist < min_dist:
                    min_dist = dist
                    winner_pos = (i, j)
        return winner_pos

    def upd_weight(self, input_vec, winner_pos):
        winner_x, winner_y = winner_pos
        self.iteration += 1  # Incrementar iteración
        current_lr = self.learning_rate / self.iteration  # η(i) = 1/i  Calcula el nuevo learning_rate decayendo ,es de la clase la formula ,matematicamente da bien pero nose si es optimo
        current_neighborhood_ratio = max(1, self.neighborhood_ratio * np.exp(-self.iteration / 2000)) # Calculo el nuevo radio  con función exponencial decreciente, porque la de la clase es x=x-1 
        for i in range(self.k):
            for j in range(self.k):
                grid_dist = np.sqrt((i - winner_x)**2 + (j - winner_y)**2)
                if grid_dist <= current_neighborhood_ratio:
                    self._update_single_neuron(i, j, input_vec, current_lr)
             
    def _update_single_neuron(self, i, j, input_vec, current_lr):
        current_weights = self.map[i, j]
        weight_difference = input_vec - current_weights
        self.map[i, j] = current_weights + current_lr * weight_difference # W_j^(i+1) = W_j^i + η(i) * (X^p - W_j^i)
        
    def train(self, X, epochs):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples) #Devuelve un array de indices aleatorios
            for idx in indices:
                input_vec = X[idx] #Obtiene una fila especifica del dataset
                winner_pos = self.winner_neuron(input_vec)
                self.upd_weight(input_vec, winner_pos)
    
         
            if (epoch + 1) % 1 == 0:                      # Mostrar progreso cada época
                current_lr = self.learning_rate / self.iteration
                current_radio = max(1, self.neighborhood_ratio * np.exp(-self.iteration / 2000))
                print(f"Época {epoch + 1}/{epochs}, lr={current_lr:.6f}, radio={current_radio:.2f}")

        print("Fin del entrenamiento")



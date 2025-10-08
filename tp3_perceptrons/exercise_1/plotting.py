import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(perceptron, function_name):
    if not perceptron.training_history:
        print("No hay historial de entrenamiento para graficar")
        return
    
    epochs = [entry['epoch'] for entry in perceptron.training_history]
    errors = [entry['errors'] for entry in perceptron.training_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title(f'Errores vs Época - Función {function_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Número de Errores', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_decision_lines_evolution(perceptron, X, y, function_name):
    if not perceptron.training_history:
        print("No hay historial de entrenamiento para mostrar el hiperplano de separación")
        return
    
    # Convertir datos a arrays de numpy
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Rango para los ejes
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    
    epochs_to_show = perceptron.training_history
    
    previous_weights = None
    tolerance = 0.001  # Tolerancia para considerar que los pesos cambiaron
    
    for i, epoch_data in enumerate(epochs_to_show):
        w1, w2 = epoch_data['weights']
        bias = epoch_data['bias']
        epoch_num = epoch_data['epoch']
        errors = epoch_data['errors']
        
        # Verificar si los pesos cambiaron significativamente
        weights_changed = True
        if previous_weights is not None:
            w1_prev, w2_prev, bias_prev = previous_weights
            if (abs(w1 - w1_prev) < tolerance and 
                abs(w2 - w2_prev) < tolerance and 
                abs(bias - bias_prev) < tolerance):
                weights_changed = False
        
        # Solo mostrar gráfico si los pesos cambiaron
        if weights_changed:
            # Calcular errores reales con los pesos actuales
            real_errors = 0
            for inputs, target in zip(X, y):
                # Calcular suma ponderada
                weighted_sum = w1 * inputs[0] + w2 * inputs[1] + bias
                # Aplicar función step
                prediction = 1 if weighted_sum >= 0 else -1
                if prediction != target:
                    real_errors += 1
            
            plt.figure(figsize=(10, 8))
            
            # Graficar puntos de datos
            colors = ['red' if label == -1 else 'blue' for label in y_array]
            plt.scatter(X_array[:, 0], X_array[:, 1], c=colors, s=100, edgecolors='black', linewidth=2)
            
            # Graficar recta de decisión
            if w2 != 0:
                x_line = np.linspace(x_min, x_max, 100)
                y_line = -(w1 * x_line + bias) / w2
                mask = (y_line >= y_min) & (y_line <= y_max)
                plt.plot(x_line[mask], y_line[mask], 'black', linewidth=3, 
                        label=f'Hiperplano: {w1:.2f}x₁ + {w2:.2f}x₂ + {bias:.2f} = 0')
            
            plt.title(f'Hiperplano de Separación - Época {epoch_num} - Función {function_name}\nErrores: {real_errors}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('x₁', fontsize=12)
            plt.ylabel('x₂', fontsize=12)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Actualizar pesos anteriores
            previous_weights = (w1, w2, bias)
    
    print(f"Evolución de rectas de decisión completada para {function_name}.")

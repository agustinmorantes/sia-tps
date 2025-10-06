from typing import List
from perceptron import SimplePerceptron
from plotting import plot_training_history, plot_decision_lines_evolution

def test_logical_function(name: str, X: List[List[float]], y: List[int]):

    print(f"\nPrueba con: {name}\n")
    
    perceptron = SimplePerceptron(learning_rate=0.01, max_epochs=200)
    
    print("Datos de entrenamiento:")
    for i, (inputs, target) in enumerate(zip(X, y)):
        print(f"  {inputs} -> {target}")
    
    print("\nIniciando entrenamiento...")
    converged = perceptron.train(X, y)
    
    if converged:
        print("\nEvaluando perceptrón entrenado:")
        accuracy, predictions = perceptron.evaluate(X, y)
        
        print(f"\nResultados:")
        print(f"  Precisión: {accuracy:.2%}")
        print(f"  Predicciones: {predictions}")
        print(f"  Objetivos:   {y}")
        
       
        # Crear gráficos
        plot_decision_lines_evolution(perceptron, X, y, name)
        plot_training_history(perceptron, name)
        
    else:
        print(f"\nEl perceptrón simple no pudo aprender la función {name}")
        plot_training_history(perceptron, name)

def main():
    
    X_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_and = [-1, -1, -1, 1]
    
    X_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_xor = [1, 1, -1, -1]
    
    # Probar función AND
    test_logical_function("AND", X_and, y_and)
    
    # Probar función XOR
    test_logical_function("XOR", X_xor, y_xor)
    
if __name__ == "__main__":
    main()

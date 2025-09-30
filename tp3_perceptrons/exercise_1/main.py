from typing import List
from perceptron import SimplePerceptron

def test_logical_function(name: str, X: List[List[float]], y: List[int]):

    print(f"\nPrueba con: {name}\n")
    
    perceptron = SimplePerceptron(learning_rate=0.1, max_epochs=1000)
    
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
        
        # Mostrar predicciones detalladas
        print(f"\nPredicciones detalladas:")
        for i, (inputs, target, prediction) in enumerate(zip(X, y, predictions)):
            status = "OK" if prediction == target else "WRONG"
            print(f"  {inputs} -> {prediction} (esperado: {target}) {status}")
    else:
        print(f"\nEl perceptrón simple no pudo aprender la función {name}")

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

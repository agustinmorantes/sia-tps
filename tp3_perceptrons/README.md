# TP3 - Perceptrones y Redes Neuronales

Este proyecto implementa diferentes tipos de perceptrones y redes neuronales para resolver problemas de clasificación. Contiene 4 ejercicios que van desde perceptrones simples hasta redes multicapa para clasificación de dígitos MNIST.

##  Ejecución de Ejercicios

### **Ejercicio 1: Perceptrón Simple**
Implementa un perceptrón simple para funciones lógicas (AND, XOR).

```bash
cd exercise_1
python main.py
```

### **Ejercicio 2: Perceptrón Multicapa**
Compara diferentes arquitecturas y funciones de activación.

```bash
cd exercise_2
python main.py config/ej2_linear.json
python main.py config/ej2_logistic.json
python main.py config/ej2_tanh.json
```

### **Ejercicio 3: Redes Neuronales Avanzadas**
Implementa redes neuronales para clasificación de dígitos manuscritos.

```bash
cd exercise_3
python main.py config/exercise_3_1.json
python main.py config/exercise_3_2.json
python main.py config/exercise_3_3.json
```

### **Ejercicio 4: Clasificación MNIST**
Implementa un clasificador completo para el dataset MNIST.

```bash
cd exercise_4
python main.py config/config.json
```

## Estructura del Proyecto

```
tp3_perceptrons/
├── exercise_1/                 
│   ├── main.py
│   ├── perceptron.py
│   └── plotting.py
├── exercise_2/                  
│   ├── main.py
│   ├── config/
│   ├── src/exercise2/
│   └── resources/
├── exercise_3/                  
│   ├── main.py
│   ├── config/
│   ├── exercise_3_1.py
│   ├── exercise_3_2.py
│   ├── exercise_3_3.py
│   └── results_3_2/
├── exercise_4/                  
│   ├── main.py
│   ├── mnist_classifier.py
│   ├── mnist_loader.py
│   ├── visualization.py
│   ├── config/
│   └── resources/
├── src/
│   └── multi_layer_perceptron.py
├── requirements.txt
└── README.md
```

**Nota**: Cada ejercicio es independiente y puede ejecutarse por separado. Los resultados se guardan automáticamente en las carpetas correspondientes.

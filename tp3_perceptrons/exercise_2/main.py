import sys
import json
import os
from src.exercise2.training_config import PerceptronTrainingConfig
from src.exercise2.perceptron_comparison import PerceptronComparison


def main():
    
    if len(sys.argv) != 2:
        print("Uso: python main.py <archivo_config>")
        print("Ejemplo: python main.py config/config.json")
        sys.exit(1)

    config_file_path = sys.argv[1]
    
    try:
        # Cargar configuración
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        # Inicializar configuración de entrenamiento
        PerceptronTrainingConfig(
            epsilon=config["epsilon"],
            seed=config["seed"],
            maxEpochs=config["maxEpochs"]
        )

        # Configurar ruta del dataset
        dataset_path = os.path.join(
            os.path.dirname(sys.argv[0]), 
            "resources", 
            "set.csv"
        )

        # Crear y ejecutar comparación de perceptrones
        comparison = PerceptronComparison(dataset_path, config)
        comparison.compare_perceptrons()

    except FileNotFoundError:
        print(f"Error: Archivo de configuración '{config_file_path}' no encontrado.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: JSON inválido en el archivo de configuración '{config_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
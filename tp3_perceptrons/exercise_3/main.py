import sys
import json
import os
import importlib
import traceback

def run_single_experiment(module_name, config, experiment_name=None):
    """Ejecuta un experimento individual dado el módulo y la configuración."""
    try:
        # Importar dinámicamente el módulo (e.g. "exercise_3.exercise_3_2")
        exercise_module = importlib.import_module(module_name)

        # Log bonito
        if experiment_name:
            print(f"\n🚀 Ejecutando experimento: {experiment_name}")
        else:
            print(f"\n🚀 Ejecutando módulo: {module_name}")

        # Si es exercise_3_3 y tiene nombre → pasar 2 argumentos
        if experiment_name and module_name == "exercise_3.exercise_3_3":
            exercise_module.main(config, experiment_name)
        else:
            exercise_module.main(config)

        print(f"✅ Experimento finalizado correctamente: {experiment_name or module_name}")
        return True

    except ImportError as e:
        print(f"❌ Error importando el módulo '{module_name}': {e}")
        return False
    except AttributeError:
        print(f"❌ El módulo '{module_name}' no tiene una función 'main'.")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando el experimento '{experiment_name or module_name}': {e}")
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <ruta_config.json>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"❌ No existe el archivo de configuración: {config_file_path}")
        sys.exit(1)

    # 📌 Agregar la raíz del proyecto al path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    try:
        with open(config_file_path, "r") as f:
            full_config = json.load(f)

        # 🧪 Múltiples experimentos
        if isinstance(full_config, list):
            total = len(full_config)
            success = 0
            print(f"📦 Se encontraron {total} experimentos en {config_file_path}")

            for i, experiment in enumerate(full_config, 1):
                module_name = experiment.get("module")
                config = experiment.get("config", {})
                name = experiment.get("name", f"Experiment {i}")

                print("\n" + "=" * 60)
                print(f"📊 Experimento {i}/{total}: {name}")
                print("=" * 60)

                if run_single_experiment(module_name, config, name):
                    success += 1

            print("\n" + "=" * 60)
            print(f"✅ {success}/{total} experimentos finalizados con éxito")
            print("=" * 60)

        # 🧪 Un solo experimento
        elif isinstance(full_config, dict):
            module_name = full_config.get("module")
            config = full_config.get("config", {})

            if not module_name:
                print("❌ No se encontró el campo 'module' en el archivo de configuración.")
                sys.exit(1)

            if not run_single_experiment(module_name, config):
                sys.exit(1)

        else:
            print("❌ Formato de configuración no válido (debe ser lista o diccionario).")
            sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"❌ Error parseando JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

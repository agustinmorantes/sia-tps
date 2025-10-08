import sys
import json
import os
import importlib
from datetime import datetime

def run_experiment(name, module_name, config, run_id):
    """Ejecuta un experimento individual"""
    try:
        exercise_module = importlib.import_module(module_name)
        cfg = dict(config)
        cfg["run_name"] = name
        print(f"\n🚀 [RUN {run_id}] Ejecutando experimento '{name}' con config:")
        print(json.dumps(cfg, indent=2))
        exercise_module.main(cfg)
    except ImportError as e:
        print(f"❌ Error importando módulo '{module_name}': {e}")
    except AttributeError:
        print(f"❌ El módulo '{module_name}' no tiene una función 'main(config)'.")
    except Exception as e:
        print(f"❌ Error ejecutando experimento {run_id} ({name}): {e}")

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <ruta_config.json>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: No se encontró el archivo de configuración '{config_file_path}'.")
        sys.exit(1)

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    with open(config_file_path, 'r') as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        print("Error: el archivo JSON debe contener una lista de configuraciones.")
        sys.exit(1)

    results_dir = os.path.join(os.path.dirname(config_file_path), "results_3_2")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary_results.json")

    all_results = []

    print(f"📊 Ejecutando {len(configs)} configuraciones...\n")

    for i, entry in enumerate(configs, start=1):
        name = entry.get("name", f"experiment_{i}")
        module_name = entry.get("module")
        config = entry.get("config")

        if not module_name or not config:
            print(f"⚠️ Experimento {i} ({name}) omitido: falta 'module' o 'config'.")
            continue

        run_experiment(name, module_name, config, i)

        all_results.append({
            "run": i,
            "name": name,
            "module": module_name,
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Todos los experimentos finalizados.")
    print(f"📁 Resultados guardados en: {summary_path}")

if __name__ == "__main__":
    main()

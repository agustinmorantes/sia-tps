import sys
import json
import os
import importlib

def run_single_experiment(module_name, config, experiment_name=None):
    """Run a single experiment"""
    try:
        exercise_module = importlib.import_module(module_name)
        if experiment_name:
            print(f"\nRunning experiment: {experiment_name}")
            if module_name == "exercise_3_3":
                exercise_module.main(config, experiment_name)
                return True

        exercise_module.main(config)
        return True
    except ImportError as e:
        print(f"Error importing module '{module_name}': {e}")
        return False
    except AttributeError:
        print(f"Module '{module_name}' does not have a 'main' function.")
        return False
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    # Check if config file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Config file '{config_file_path}' not found.")
        sys.exit(1)

    # Add parent directory to Python path so exercise modules can import src
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        # Load the configuration
        with open(config_file_path, 'r') as f:
            full_config = json.load(f)

        # Check if it's a multi-experiment config (list) or single experiment (dict)
        if isinstance(full_config, list):
            # Multi-experiment config
            print(f"Found {len(full_config)} experiments to run")
            success_count = 0

            for i, experiment in enumerate(full_config, 1):
                module_name = experiment.get('module')
                config = experiment.get('config', {})
                name = experiment.get('name', f'Experiment {i}')

                if not module_name:
                    print(f"Skipping experiment {i}: 'module' field not found")
                    continue

                print(f"\n{'='*60}")
                print(f"Experiment {i}/{len(full_config)}: {name}")
                print(f"{'='*60}")

                if run_single_experiment(module_name, config, name):
                    success_count += 1

            print(f"\n{'='*60}")
            print(f"Completed {success_count}/{len(full_config)} experiments successfully")
            print(f"{'='*60}")

        else:
            # Single experiment config
            module_name = full_config.get('module')
            config = full_config.get('config', {})

            if not module_name:
                print("Error: 'module' field not found in config file.")
                sys.exit(1)

            if not run_single_experiment(module_name, config):
                sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

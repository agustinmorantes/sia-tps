import sys
import json
import os
import importlib

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

        # Extract module name and config
        module_name = full_config.get('module')
        config = full_config.get('config', {})

        if not module_name:
            print("Error: 'module' field not found in config file.")
            sys.exit(1)

        # Import and run the appropriate exercise
        try:
            exercise_module = importlib.import_module(module_name)
            exercise_module.main(config)
        except ImportError as e:
            print(f"Error importing module '{module_name}': {e}")
            sys.exit(1)
        except AttributeError:
            print(f"Error: Module '{module_name}' does not have a 'main' function.")
            sys.exit(1)
        except Exception as e:
            print(f"Error running exercise: {e}")
            sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import subprocess
from pathlib import Path

def run_all_configs():
    """Run main.py for each config file in the configs directory"""
    configs_dir = Path("configs")

    if not configs_dir.exists():
        print("Error: configs/ directory not found")
        return

    # Get all JSON files in configs directory
    config_files = sorted(configs_dir.glob("*.json"))

    if not config_files:
        print("No config files found in configs/ directory")
        return

    print("="*60)
    print(f"Found {len(config_files)} configuration files")
    print("="*60)

    for idx, config_file in enumerate(config_files, 1):
        print(f"\n[{idx}/{len(config_files)}] Running: {config_file.name}")
        print("-"*60)

        try:
            # Run main.py with the config file
            result = subprocess.run(
                ["python", "main.py", str(config_file)],
                capture_output=True,
                text=True,
                check=True
            )

            # Print output if any
            if result.stdout:
                print(result.stdout)

            print(f"✓ Successfully processed {config_file.name}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {config_file.name}")
            print(f"Error message: {e.stderr}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

    print("\n" + "="*60)
    print("All configurations processed!")
    print("="*60)

if __name__ == "__main__":
    run_all_configs()


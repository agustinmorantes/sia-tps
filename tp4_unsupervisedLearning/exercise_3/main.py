import json
import os.path
import sys

import Hopfield

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    hopfield, query_pattern, crosstalk, states, energies = Hopfield.main(config)
    metrics = {
        "num_iterations": len(states),
        "energies": energies,
        "crosstalk": crosstalk.flatten().tolist(),
        "states": [s.flatten().tolist() for s in states],
        "query_pattern": query_pattern.flatten().tolist()
    }

    os.makedirs("metrics", exist_ok=True)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    with open(f"metrics/{config_name}.json", 'w') as f:
        json.dump(metrics, f)

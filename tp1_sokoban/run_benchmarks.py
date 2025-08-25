
import os
import sys
import importlib
from src.classes.Point import Point
from src.classes.Sokoban import SokobanManager
from src.map_parser import load_and_parse_map

HEURISTICS_DIR = "src.classes.heuristics"

def get_all_heuristics():
    heuristics = []
    for filename in os.listdir(os.path.join("src", "classes", "heuristics")):
        if filename.endswith("Heuristic.py") and filename != "Heuristic.py" and filename != "__init__.py":
            module_name = filename[:-3]
            module = importlib.import_module(f"{HEURISTICS_DIR}.{module_name}")
            heuristic_class = getattr(module, module_name)
            heuristics.append(heuristic_class())
    return heuristics

def run_benchmark(map_file, algorithm_name, sokoban_manager, heuristic=None):
    print(f"Running {algorithm_name} on {map_file}...")
    sokoban_manager.reset()
    solution_path = None
    if algorithm_name == "bfs":
        solution_path = sokoban_manager.bfs()
    elif algorithm_name == "dfs":
        solution_path = sokoban_manager.dfs()
    elif algorithm_name == "greedy":
        solution_path = sokoban_manager.greedy(heuristic)
    elif algorithm_name == "a_star":
        solution_path = sokoban_manager.a_star(heuristic)

    if solution_path:
        print(f"  Solución encontrada en {len(solution_path) - 1} movimientos!")
        print(f"  Nodos expandidos: {sokoban_manager.nodes_expanded}")
        print(f"  Tiempo de ejecución (s): {sokoban_manager.execution_time}")
        return {
            "solution_cost": len(solution_path) - 1,
            "nodes_expanded": sokoban_manager.nodes_expanded,
            "border_nodes_count": sokoban_manager.border_nodes_count,
            "execution_time": sokoban_manager.execution_time
        }
    else:
        print("  No se encontró solución.")
        return {"solution_cost": -1, "nodes_expanded": -1, "border_nodes_count": -1, "execution_time": -1}

def main():
    maps_dir = "maps"
    map_files = [os.path.join(maps_dir, f) for f in os.listdir(maps_dir) if f.endswith(".txt")]

    heuristics = get_all_heuristics()

    for map_file in map_files:
        map_data, walls, goals, boxes, player_pos, size = load_and_parse_map(map_file)
        player = Point(*player_pos)
        boxes_points = {Point(x, y) for x, y in boxes}
        walls_points = {Point(x, y) for x, y in walls}
        goals_points = {Point(x, y) for x, y in goals}

        sokoban = SokobanManager(walls=walls_points, goals=goals_points, player=player, boxes=boxes_points, size=size)

        results = {}
        map_name = os.path.basename(map_file).replace(".txt", "")

        # BFS
        results["bfs"] = run_benchmark(map_file, "bfs", sokoban)

        # DFS
        results["dfs"] = run_benchmark(map_file, "dfs", sokoban)

        # Greedy and A* with all heuristics
        for heuristic in heuristics:
            heuristic_name = heuristic.__class__.__name__.replace("Heuristic", "")
            results[f"greedy_{heuristic_name}"] = run_benchmark(map_file, "greedy", sokoban, heuristic)
            results[f"a_star_{heuristic_name}"] = run_benchmark(map_file, "a_star", sokoban, heuristic)

        output_filename = f"results_{map_name}.txt"
        with open(output_filename, "w") as f:
            for algo, stats in results.items():
                f.write(f"{algo}: {stats}\n")
        print(f"Results for {map_name} saved to {output_filename}")

if __name__ == "__main__":
    main()

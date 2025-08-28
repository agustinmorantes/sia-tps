import os
import sys
import arcade

from src.classes.Point import Point
from src.classes.Sokoban import SokobanManager
from src.classes.heuristics.ChebyshevHeuristic import ChebyshevHeuristic
from src.classes.heuristics.EuclideanHeuristic import EuclideanHeuristic
from src.classes.heuristics.HammingHeuristic import HammingHeuristic
from src.classes.heuristics.ManhattanHeuristic import ManhattanHeuristic
from src.classes.heuristics.ManhattanImprovedHeuristic import ManhattanImprovedHeuristic
from src.map_parser import load_and_parse_map
from src.map_viewer import SokobanMapViewer

heuristics = {
    "chebyshev": ChebyshevHeuristic(),
    "euclidean": EuclideanHeuristic(),
    "hamming": HammingHeuristic(),
    "manhattan": ManhattanHeuristic(),
    "manhattan_improved": ManhattanImprovedHeuristic()
}

def main():
    map_file = ""
    algorithm_name = ""
    heuristic_name = ""

    if len(sys.argv) > 2:
        map_file = sys.argv[1]
        algorithm_name = sys.argv[2]

        if len(sys.argv) > 3:
            heuristic_name = sys.argv[3]

    else:
        print("Uso: python main.py [map_file] [algorithm_name] [heuristic_name]."
              "Algoritmos: dfs, bfs, greedy, astar"
              "Heurísticas: chebyshev, euclidean, hamming, manhattan, manhattan_improved")
        exit(1)

    if not os.path.exists(map_file):
        print(f"Error: el archivo de mapa no se encontró en {map_file}")
        exit(1)

    if algorithm_name not in ["dfs", "bfs", "greedy", "astar"]:
        print("Error: Algoritmo inválido. Opciones: dfs, bfs, greedy, astar")
        exit(1)

    if algorithm_name in ["greedy", "astar"] and not heuristic_name in heuristics:
        print(f"Error: Heurística inválida. Opciones: {', '.join(heuristics.keys())}")
        exit(1)

    map_data, walls, goals, boxes, player_pos, size = load_and_parse_map(map_file)

    player = Point(*player_pos)
    boxes_points = {Point(x, y) for x, y in boxes}
    walls_points = {Point(x, y) for x, y in walls}
    goals_points = {Point(x, y) for x, y in goals}

    sokoban = SokobanManager(walls=walls_points, goals=goals_points, player=player, boxes=boxes_points, size=size)


    solution_path = None

    if algorithm_name == "dfs":
        solution_path = sokoban.dfs()
    elif algorithm_name == "bfs":
        solution_path = sokoban.bfs()
    elif algorithm_name == "greedy":
        heuristic = heuristics[heuristic_name]
        solution_path = sokoban.greedy(heuristic)
    elif algorithm_name == "astar":
        heuristic = heuristics[heuristic_name]
        solution_path = sokoban.a_star(heuristic)

    if solution_path:
        print(f"Solución encontrada en {len(solution_path)-1} movimientos!")
        print("Estados del camino:")
        # for state in solution_path:
        #     print(state)  # Podés definir __str__ en State para mostrar más claro
        print("Nodos expandidos:", sokoban.nodes_expanded)
        print("Border nodes máximo:", sokoban.border_nodes_count)
        print("Tiempo de ejecución (s):", sokoban.execution_time)
        game = SokobanMapViewer(map_file, solution_path)
        arcade.run()
    else:
        print("No se encontró solución.")

if __name__ == "__main__":
    main()

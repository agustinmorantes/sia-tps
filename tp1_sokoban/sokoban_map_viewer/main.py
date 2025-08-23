import os
import sys
from collections import deque
from classes.Point import Point
from classes.Sokoban import SokobanManager
from map_parser import load_and_parse_map
from map_viewer import SokobanMapViewer
import arcade

from sokoban_map_viewer.classes.heuristics.ChebyshevHeuristic import ChebyshevHeuristic
from sokoban_map_viewer.classes.heuristics.EuclideanHeuristic import EuclideanHeuristic
from sokoban_map_viewer.classes.heuristics.HammingHeuristic import HammingHeuristic
from sokoban_map_viewer.classes.heuristics.ManhattanHeuristic import ManhattanHeuristic
from sokoban_map_viewer.classes.heuristics.ManhattanImprovedHeuristic import ManhattanImprovedHeuristic


def main():
    map_file = os.path.join(f"{os.path.abspath(__file__)}","maps", "Map1.txt")
    if len(sys.argv) > 1:
        map_file = sys.argv[1]

    if not os.path.exists(map_file):
        print(f"Error: el archivo de mapa no se encontró en {map_file}")
        return

    map_data, walls, goals, boxes, player_pos = load_and_parse_map(map_file)

    player = Point(*player_pos)
    boxes_points = {Point(x, y) for x, y in boxes}
    walls_points = {Point(x, y) for x, y in walls}
    goals_points = {Point(x, y) for x, y in goals}

    sokoban = SokobanManager(board=walls_points, goals=goals_points, player=player, boxes=boxes_points)

    solution_path = sokoban.a_star(HammingHeuristic())

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

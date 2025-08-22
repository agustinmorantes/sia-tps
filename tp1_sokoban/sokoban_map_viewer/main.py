import os
from collections import deque
from classes.Point import Point
from classes.Sokoban import SokobanManager
from map_parser import load_and_parse_map

def main():
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    map_file = os.path.join(maps_dir, "Map1.txt")  # Cambiar por tu mapa

    if not os.path.exists(map_file):
        print(f"Error: el archivo de mapa no se encontró en {map_file}")
        return

    map_data, walls, goals, boxes, player_pos = load_and_parse_map(map_file)

    player = Point(*player_pos)
    boxes_points = {Point(x, y) for x, y in boxes}
    walls_points = {Point(x, y) for x, y in walls}
    goals_points = {Point(x, y) for x, y in goals}

    sokoban = SokobanManager(board=walls_points, goals=goals_points, player=player, boxes=boxes_points)

    solution_path = sokoban.bfs()

    if solution_path:
        print(f"Solución encontrada en {len(solution_path)-1} movimientos!")
        print("Estados del camino:")
        for state in solution_path:
            print(state)  # Podés definir __str__ en State para mostrar más claro
        print("Nodos expandidos:", sokoban.nodes_expanded)
        print("Border nodes máximo:", sokoban.border_nodes_count)
        print("Tiempo de ejecución (s):", sokoban.execution_time)
    else:
        print("No se encontró solución.")

if __name__ == "__main__":
    main()

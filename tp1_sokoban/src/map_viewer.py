import arcade
from .map_parser import load_and_parse_map
import os
from .classes.State import State
from .classes.Point import Point

SCREEN_TITLE = "Sokoban Map Viewer"
PNG_SIZE = 64

ELEMENTS = {
    "#": "./textures/wall.png",   # Pared
    " ": "./textures/free.png",   # Espacio libre (null)
    ".": "./textures/goal.png",   # Objetivo
    "@": "./textures/player.png", # Posicion inicial del jugador
    "$": "./textures/box.png",    # Caja
    "*": "./textures/box_goal.png" # Caja sobre objetivo
}

class SokobanMapViewer(arcade.Window):
    def __init__(self, map_file_path, solution_path=None):
        self.map_data, walls, goals, boxes, player_position, _ = load_and_parse_map(map_file_path)

        self.walls = {Point(r, c) for r, c in walls}
        self.goals = {Point(r, c) for r, c in goals}
        self.boxes_initial = {Point(r, c) for r, c in boxes}
        self.player_position_initial = Point(*player_position)

        self.num_rows = len(self.map_data)
        self.num_cols = len(self.map_data[0])
        width = self.num_cols * PNG_SIZE
        height = self.num_rows * PNG_SIZE

        super().__init__(width, height, SCREEN_TITLE)   
        arcade.set_background_color(arcade.color.ANTIQUE_BRONZE)

        self.textures = {
            key: arcade.load_texture(img) if img else None
            for key, img in ELEMENTS.items()
        }

        self.solution_path = solution_path
        self.current_state_index = 0
        self.current_state = None
        if self.solution_path:
            self.current_state = self.solution_path[self.current_state_index]
            arcade.schedule(self.update_game_state, min(0.5, 10.0 / len(solution_path)))  # Update every 0.5 seconds or faster for long solutions
        
    def update_game_state(self, delta_time):
        if self.solution_path and self.current_state_index < len(self.solution_path) - 1:
            self.current_state_index += 1
            self.current_state = self.solution_path[self.current_state_index]
        else:
            arcade.unschedule(self.update_game_state)

    def on_draw(self):
        self.clear()

        if not self.current_state: # If no solution is loaded, draw the initial map
            # Draw floor and goals first
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    x = col * PNG_SIZE + PNG_SIZE // 2
                    y = (self.num_rows - row - 1) * PNG_SIZE + PNG_SIZE // 2  
    
                    point = Point(row, col)
                    if point in self.walls:
                        continue  # Walls are drawn later
                    elif point in self.goals:
                        arcade.draw_texture_rect(self.textures["."], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
                    else:
                        arcade.draw_texture_rect(self.textures[" "], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
            
            # Draw walls
            for wall_point in self.walls:
                x = wall_point.y * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - wall_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
                arcade.draw_texture_rect(self.textures["#"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
            
            # Draw boxes
            for box_point in self.boxes_initial:
                x = box_point.y * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - box_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
                if box_point in self.goals:
                    arcade.draw_texture_rect(self.textures["*"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
                else:
                    arcade.draw_texture_rect(self.textures["$"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
            
            # Draw player
            player_point = self.player_position_initial
            x = player_point.y * PNG_SIZE + PNG_SIZE // 2
            y = (self.num_rows - player_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
            arcade.draw_texture_rect(self.textures["@"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
        else: # Draw the current state from the solution path
            # Draw floor and goals first
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    x = col * PNG_SIZE + PNG_SIZE // 2
                    y = (self.num_rows - row - 1) * PNG_SIZE + PNG_SIZE // 2
                    
                    point = Point(row, col)
                    if point in self.walls:
                        continue  # Walls are drawn later
                    elif point in self.goals:
                        arcade.draw_texture_rect(self.textures["."], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
                    else:
                        arcade.draw_texture_rect(self.textures[" "], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))

            # Draw walls
            for wall_point in self.walls:
                x = wall_point.y * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - wall_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
                arcade.draw_texture_rect(self.textures["#"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))

            # Draw boxes
            for box_point in self.current_state.boxes:
                x = box_point.y * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - box_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
                if box_point in self.goals:
                    arcade.draw_texture_rect(self.textures["*"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
                else:
                    arcade.draw_texture_rect(self.textures["$"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))

            # Draw player
            player_point = self.current_state.player
            x = player_point.y * PNG_SIZE + PNG_SIZE // 2
            y = (self.num_rows - player_point.x - 1) * PNG_SIZE + PNG_SIZE // 2
            arcade.draw_texture_rect(self.textures["@"], arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y))
    
if __name__ == "__main__":
 
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    map_file = os.path.join(maps_dir, "Map3.txt") 

    if not os.path.exists(map_file):
        print(f"Error: El archivo de mapa no se encontró en {map_file}")
    else:
        game = SokobanMapViewer(map_file)
        arcade.run()

import arcade
from map_parser import load_and_parse_map
import os

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
    def __init__(self, map_file_path):
        self.map_data, self.walls, self.goals, self.boxes, self.player_position = load_and_parse_map(map_file_path)

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

    def on_draw(self):
        self.clear()

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - row - 1) * PNG_SIZE + PNG_SIZE // 2  

                tile_char = self.map_data[row][col]
                
                # Draw base tile (free space or goal)
                if tile_char in self.textures and self.textures[tile_char]:
                    rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                    arcade.draw_texture_rect(self.textures[tile_char], rect)

        # Draw walls, boxes, and player on top
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * PNG_SIZE + PNG_SIZE // 2
                y = (self.num_rows - row - 1) * PNG_SIZE + PNG_SIZE // 2  

                current_char = self.map_data[row][col]

                if current_char == '#': # Wall
                    rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                    arcade.draw_texture_rect(self.textures["#"], rect)
                elif [row, col] in self.boxes: # Box
                    if [row, col] in self.goals: # Box on goal
                        rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                        arcade.draw_texture_rect(self.textures["*"], rect)
                    else: # Regular box
                        rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                        arcade.draw_texture_rect(self.textures["$"], rect)
                elif [row, col] == self.player_position: # Player
                    rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                    arcade.draw_texture_rect(self.textures["@"], rect)
                elif [row, col] in self.goals: # Goal (if not covered by box)
                     rect = arcade.Rect(x-PNG_SIZE//2, x+PNG_SIZE//2, y-PNG_SIZE//2, y+PNG_SIZE//2, PNG_SIZE, PNG_SIZE, x, y)
                     arcade.draw_texture_rect(self.textures["."], rect)
                

if __name__ == "__main__":
 
    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    map_file = os.path.join(maps_dir, "Map3.txt") 

    if not os.path.exists(map_file):
        print(f"Error: El archivo de mapa no se encontró en {map_file}")
    else:
        game = SokobanMapViewer(map_file)
        arcade.run()

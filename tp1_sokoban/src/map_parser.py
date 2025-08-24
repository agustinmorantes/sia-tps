import os

def load_and_parse_map(map_file_path):
    with open(map_file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    max_length = max(len(line) for line in lines)
    map_data = [list(line.ljust(max_length)) for line in lines]

    walls = [] 
    goals = []
    boxes = []
    player_position = [0, 0]

    for row in range(0, len(map_data)):
        for col in range(0, len(map_data[0])):
            current_element = map_data[row][col]
            match current_element:
                case '#': 
                    walls.append([row, col])
                case '$':
                    boxes.append([row, col])
                case '@':
                    player_position = [row,col] #Lo voy a dibujar con player_position entonces lo borro de aca porque esto se dibuja completo luego
                    map_data[row][col] = ' ' 
                case '.':
                    goals.append([row, col])
                case '*': 
                    boxes.append([row , col])
                    goals.append([row, col]) 

    return map_data, walls, goals, boxes, player_position

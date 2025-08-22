from collections import deque
from Node import Node
from State import State

class SokobanManager:
    _instance = None
    
    def __new__(cls, board, goals, player, boxes, deadlocks=set()):
        if cls._instance is None:
            cls._instance = super(SokobanManager, cls).__new__(cls)
            cls._instance._initialize(board, goals, player, boxes, deadlocks)
        return cls._instance

    def _initialize(self, board, goals, player, boxes, deadlocks):
        self.initial_state = State(player, set(boxes), set(board), set(goals), set(deadlocks))
        self.root_node = Node(self.initial_state)
        self.visited_nodes = set()
        self.winning_path = deque()
        self.border_nodes_count = 0
        self.nodes_expanded = 0
        self.solution_cost = 0
        self.execution_time = 0
        self.start_time = 0
        self.heuristics = {}

    
    

from collections import deque
from classes.Node import Node
from classes.State import State
from classes.Point import Point
from classes.Direction import Direction
import time

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

    def reset(self):
        self.visited_nodes.clear()
        self.winning_path.clear()
        self.border_nodes_count = 0
        self.nodes_expanded = 0
        self.solution_cost = 0
        self.execution_time = 0
        self.start_time = 0
        self.root_node = Node(self.initial_state())
    
    def reconstruct_path(self, node: Node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def bfs(self):
        self.start_time = time.time()
        queue = deque([self.root_node])
        self.visited_nodes.add(self.root_node)

        while queue:
            current_node = queue.popleft()
            self.nodes_expanded += 1
            
            if current_node.state.is_solved():
                self.winning_path = self.reconstruct_path(current_node)
                self.solution_cost = len(self.winning_path) - 1;
                self.execution_time = time.time() - self.start_time
                return self.winning_path
            
            for child_node in current_node.get_children():
                if child_node not in self.visited_nodes:
                    self.visited_nodes.add(child_node)
                    queue.append(child_node)
            
            self.border_nodes_count = max(self.border_nodes_count, len(queue))
            
        self.execution_time = time.time() - self.start_time
        return None

    def dfs(self):
        return None
    
    def greedy(self):
        return None
    
    def a_star(self):
        return None

    def get_statistics(self):
        return {
            "nodes_expanded": self.nodes_expanded,
            "solution_cost": self.solution_cost,
            "border_nodes_count": self.border_nodes_count,
            "execution_time": self.execution_time
        }

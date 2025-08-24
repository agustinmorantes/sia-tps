import itertools
from collections import deque
import time
import heapq

from .Node import Node
from .State import State
from .heuristics.Heuristic import Heuristic


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
        self.root_node = Node(self.initial_state)
    
    def reconstruct_path(self, node: Node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def bfs(self):#Los nodos ya generados pero aun no explorados estan en queue , voy sacando el nodo mas viejo y analizando nivel por nivel
        self.start_time = time.time()
        queue = deque([self.root_node])
        self.visited_nodes.add(self.root_node)

        while queue:
            current_node = queue.popleft()
            self.nodes_expanded += 1
            
            if current_node.state.is_solved():
                self.winning_path = self.reconstruct_path(current_node)
                self.solution_cost = len(self.winning_path) - 1
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
        self.start_time = time.time()
        stack = deque([self.root_node])

        while stack:
            current_node = stack.pop()
            self.nodes_expanded += 1
            self.visited_nodes.add(current_node)

            if current_node.state.is_solved():
                self.winning_path = self.reconstruct_path(current_node)
                self.solution_cost = len(self.winning_path) - 1
                self.execution_time = time.time() - self.start_time
                return self.winning_path

            for child_node in current_node.get_children():
                if child_node not in self.visited_nodes:
                    stack.append(child_node)

            self.border_nodes_count = max(self.border_nodes_count, len(stack))

        self.execution_time = time.time() - self.start_time
        return None

    def greedy(self, heuristic: Heuristic):
        self.start_time = time.time()

        # Priority queue, ordered by heuristic value with tie-breaking using a counter
        counter = itertools.count()
        queue = [(heuristic(self.root_node.state), next(counter), self.root_node)]

        self.visited_nodes = {self.root_node}

        while queue:
            # Get the node with the lowest heuristic value from priority queue
            _, _, current_node = heapq.heappop(queue)
            self.nodes_expanded += 1

            if current_node.state.is_solved():
                self.winning_path = self.reconstruct_path(current_node)
                self.solution_cost = len(self.winning_path) - 1
                self.execution_time = time.time() - self.start_time
                return self.winning_path

            for child_node in current_node.get_children():
                if child_node not in self.visited_nodes:
                    self.visited_nodes.add(child_node)
                    heapq.heappush(queue, (heuristic(child_node.state), next(counter), child_node))

            self.border_nodes_count = max(self.border_nodes_count, len(queue))

        self.execution_time = time.time() - self.start_time
        return None
    
    def a_star(self, heuristic: Heuristic):
        self.start_time = time.time()

        counter = itertools.count()
        openQueue = [(heuristic(self.root_node.state), next(counter), self.root_node)]
        openSet = {self.root_node}

        gScore = {self.root_node: 0} #Costo real desde el nodo inicial  hasta el nodo n 
        fScore = {self.root_node: heuristic(self.root_node.state)}

        while openQueue:
            _, _, current_node = heapq.heappop(openQueue)
            openSet.remove(current_node)

            if current_node.state.is_solved():
                self.winning_path = self.reconstruct_path(current_node)
                self.solution_cost = len(self.winning_path) - 1
                self.execution_time = time.time() - self.start_time
                return self.winning_path

            for child_node in current_node.get_children():
                tentative_gScore = gScore[current_node] + 1

                if child_node not in gScore or tentative_gScore < gScore[child_node]: #Si no tiene un costo registrado o el camino actual es mas barato
                    gScore[child_node] = tentative_gScore
                    fScore[child_node] = tentative_gScore + heuristic(child_node.state)

                    if child_node not in openSet: #Si el hijo no esta en la frontera se agrega para explorarlo mas tarde 
                        heapq.heappush(openQueue, (fScore[child_node], next(counter), child_node))
                        openSet.add(child_node)

            self.border_nodes_count = max(self.border_nodes_count, len(openQueue))
            self.nodes_expanded += 1

        self.execution_time = time.time() - self.start_time
        return None

    def get_statistics(self):
        return {
            "nodes_expanded": self.nodes_expanded,
            "solution_cost": self.solution_cost,
            "border_nodes_count": self.border_nodes_count,
            "execution_time": self.execution_time
        }

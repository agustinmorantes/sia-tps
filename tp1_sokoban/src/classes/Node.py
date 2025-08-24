from .State import State


class Node:
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children = []

    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self):
        return hash(self.state)
    
    def get_parent(self):
        return self.parent
    
    def get_state(self):
        return self.state
    
    # Ver de obtener los hijos en forma ordenadan respecto a las direcciones
    # por ejemplo siempre guardar UP DOWN LEFT RIGHT, pero probar con otros ordenes
    def get_children(self):
        if not self.children:
            for child_state in self.state.get_children():
                child_node = Node(child_state, self)
                self.children.append(child_node)
        return self.children
    
    def get_path(self):
        path = []
        current_node = self
        while current_node:
            path.append(current_node.state)
            current_node = current_node.get_parent()
        return path[::-1]

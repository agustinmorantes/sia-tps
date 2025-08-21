class Node:
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state

    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self):
        return hash(self.state)
    
    def get_parent(self):
        return self.parent
    
    def get_state(self):
        return self.state
    
    def get_children(self):
        children = []
        for child in self.state.get_children():
            children.append(Node(self, child))
        return children

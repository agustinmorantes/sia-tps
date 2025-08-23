from sokoban_map_viewer.classes.State import State

class Heuristic:
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
        self.cache = {}

    def __call__(self, state: State) -> float:
        if state in self.cache:
            return self.cache[state]

        res = self.func(state)
        self.cache[state] = res
        return res

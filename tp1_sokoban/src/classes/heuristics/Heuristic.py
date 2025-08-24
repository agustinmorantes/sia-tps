from ..State import State

class Heuristic:
    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def __call__(self, state: State) -> float:
        return self.func(state)

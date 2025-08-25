from ..State import State

class Heuristic: #devuelven una estimación mínima del costo restante ,partiendo desde el estado actual,hasta llegar al estado meta  
    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def __call__(self, state: State) -> float:
        return self.func(state)

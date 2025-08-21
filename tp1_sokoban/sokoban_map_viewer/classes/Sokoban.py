class SokobanManager:
    _instance = None
    
    def __new__(cls, board, goals, player, boxes):
        if cls._instance is None:
            cls._instance = super(SokobanManager, cls).__new__(cls)
            cls._initialize(cls, board, goals, player, boxes)
        return cls._instance

    def _initialize(self, board, goals, player, boxes):
        self.board = board
        self.goals = goals
        self.player = player
        self.boxes = boxes
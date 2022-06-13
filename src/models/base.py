from src.utils.params import Registrable


class BaseModel(Registrable):
    def __init__(self):
        super().__init__()

    def inputs(self):
        raise NotImplementedError()

    def build_graph(self):
        raise NotImplementedError()
    
    def build_graph_for_hp(self, trial):
        raise NotImplementedError()

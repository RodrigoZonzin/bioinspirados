import numpy as np


class Gene(): 
    def __init__(self, alelos: tuple()):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim

class AG(): 
    def __init__(self, pop = []): 
        self.pop = pop
        self.n_individuals  = len(self.pop)

    def print(self):
        [print(i) for i in self.pop]

    def dim_pop(self):
        return len(self.pop)

    
    
g1 = Gene((1, 0, 1))
g2 = Gene((0, 0, 1))
g3 = Gene((1, 0, 1))
g4 = Gene((0, 1, 1))
g5 = Gene((1, 1, 1))


ag = AG([g1, g2, g3, g4, g5])

ag.print()

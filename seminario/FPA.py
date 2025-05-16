import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 

class Polen(): 
    def __init__(self, dim, pmin, pmax):
        self.dim = dim
        self.valores = [random.randint(pmin, pmax) for _ in range(dim)]
        
    def __str__(self):
        return str(self.valores)

class FPA:
    def __init__(self, npop, plambda, pepsilon, dimPolen, intervalo: tuple): 
        self.npop = npop 
        self.plambda = plambda
        self.epsilon = pepsilon
        self.minvalue = intervalo[0]
        self.maxvalue = intervalo[1]
        self.dimPolen = dimPolen
        self.pop = self._gerar_pop()

    def _gerar_pop(self): 
        return np.array([Polen(self.dimPolen, self.minvalue, self.maxvalue) for _ in range(self.npop)])
    
    def bioticPollinization(self, polen): 
        novoPolen = Polen(self.dimPolen, self.maxvalue, self.maxvalue)
        novoPolen.valores[i] = polen.valores[i] + levy()*(polen.valores[i] - gestrela)

    def print_pop(self):
        [print(x) for x in self.pop]


if __name__ == 'main': 
    maxIt = 20

    fpa = FPA(10, 0.1, 0.5, 3, intervalo=(-2, 2))
    
    for t in range(maxIt):

        
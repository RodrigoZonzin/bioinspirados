import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colos as mcolors
import os 

class Polen(): 
    def __init__(self, pmin, pmax):
        self.valores = pvalores
        self.dim = 

class FPA:
    def __init__(self, dim, plambda, pepsilon, minvalue, maxvalue): 
        self.dim = dim 
        self.pop = self._gerar_pop(self.dim)
        self.plambda = plambda
        self.epsilon = pepsilon
        self.limites = (minvalue, maxvalue)

    def _gerar_pop(self, dim): 
        return np.array([Polen(self.minvalue, self.maxvalue) for ])



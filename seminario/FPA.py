import numpy as np 
import random
import pandas as pd
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 


class Polen(): 
    def __init__(self, m, min_val = -4, max_val = 4):
        self.m = m          #tamanho do vetor de solucao X = [x1, x2, ... , xm]
        self.valores = np.array([np.random.randint(min_val, max_val) for _ in range(m)]) #X

    def __str__(self):
        return ' '.join(str(v) for v in self.valores) 
    
    def __repr__(self):
        return self.__str__()

class FPA(): 
    def __init__(self, n, m = 10):
        self.intervalo = (-4, 4)
        self.n = n  
        self.m = m                                    #numero de flores na populacao
        self.pop = [Polen(self.m) for _ in range(n)]    #POP = [X1, X2, ..., Xn]
        self.L  = None 
        self.p  = 0.8
        self._lambda = 1.5
        self.maxIt = 200
        self.gestrela = None
        self.melhor_fitness = None

    def fitness(self, x):
        n = x.m
        x = x.valores
        part1 = -0.2 * np.sqrt(np.sum(x**2) / n)
        part2 = np.sum(np.cos(2 * np.pi * x)) / n
        return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e

    """
    def rodar(self): 
        results_fitness = []

        melhor_posicao = sorted([{'pos': i, 'solution': self.fitness(self.pop[i])} for i in range(self.n)], key = lambda x: x['solution'])[0]['pos']
        self.gestrela = self.pop[melhor_posicao]
        self.melhor_fitness = self.fitness(self.pop[melhor_posicao])

        print(self.gestrela)
        
        for it in range(self.maxIt): 
            for flor in self.pop: 
                
                #Polinizacao global
                if np.random.rand() < self.p: 
                    self.L = levy_stable.rvs(self._lambda, beta = 0, size=self.n)
                    for pos, polen in enumerate(self.pop): 
                        #print(polen.valores, self.gestrela)
                        polen.valores = polen.valores + self.L[pos]*(polen.valores - self.gestrela.valores)
                        polen.valores = np.clip(polen.valores, self.intervalo[0], self.intervalo[1])
                
                #Polinizacao local
                else: 
                    self.ep = np.random.uniform(size = self.m)    
                    for pos, polen in enumerate(self.pop): 
                        k = np.random.randint(*self.intervalo)
                        j = np.random.randint(*self.intervalo)
                        while k == j : 
                            j = np.random.randint(*self.intervalo)
                        
                        polen.valores = polen.valores + self.ep*(self.pop[j].valores + self.pop[k].valores)
                        polen.valores = np.clip(polen.valores, self.intervalo[0], self.intervalo[1])

        """
    def rodar(self): 
        results_fitness = []

        # Inicializa g*
        melhor_idx = min(range(self.n), key=lambda i: self.fitness(self.pop[i]))
        self.gestrela = Polen(self.m)
        self.gestrela.valores = self.pop[melhor_idx].valores.copy()
        self.melhor_fitness = self.fitness(self.gestrela)

        for it in range(self.maxIt): 
            for i, flor in enumerate(self.pop): 
                xi = flor.valores.copy()

                #Polinizacao global
                if np.random.rand() < self.p:
                    L = levy_stable.rvs(self._lambda, beta=0, size=self.m)
                    x_new = xi + L*(self.gestrela.valores - xi)
                
                #Polinizacao global
                else:
                    j = random.randint(0, self.n - 1)
                    k = random.randint(0, self.n - 1)
                    while j == k:
                        k = random.randint(0, self.n - 1)
                    epsilon = np.random.uniform(size=self.m)
                    x_new = xi + epsilon * (self.pop[j].valores - self.pop[k].valores)

                x_new = np.clip(x_new, self.intervalo[0], self.intervalo[1])

                novo_polen = Polen(self.m)
                novo_polen.valores = x_new

                if self.fitness(novo_polen) < self.fitness(flor):
                    self.pop[i].valores = x_new

                    # Atualiza g* se necessario
                    if self.fitness(novo_polen) < self.melhor_fitness:
                        self.gestrela.valores = x_new.copy()
                        self.melhor_fitness = self.fitness(novo_polen)

            results_fitness.append(self.melhor_fitness)
    
        self.historico_fitness = results_fitness

            
    def __str__(self):
        return f'Numero de polens: {self.n}\nVetor gbest:{self.gestrela}\nMelhor Fitness: {self.melhor_fitness}\n'+ str(self.pop)
        



fpa = FPA(10, m  = 3)
fpa.rodar()
plt.scatter(range(fpa.maxIt), fpa.historico_fitness)
plt.savefig('results.png', dpi = 400)
print(fpa)
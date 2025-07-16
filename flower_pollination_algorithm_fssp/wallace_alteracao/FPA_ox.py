import numpy as np 
import random
import pandas as pd
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys
from time import time


def calcular_makespan(sequencia, tempos, N, M):
    C = np.zeros((N, M))

    for i, tarefa in enumerate(sequencia):
        for m in range(M):
            tempo_proc = tempos[tarefa][m]
            if i == 0 and m == 0:
                C[i, m] = tempo_proc
            elif i == 0:
                C[i, m] = C[i, m-1] + tempo_proc
            elif m == 0:
                C[i, m] = C[i-1, m] + tempo_proc
            else:
                C[i, m] = max(C[i-1, m], C[i, m-1]) + tempo_proc

    return C[-1, -1]

class Polen():
    def __init__(self, m):
        self.m = m
        self.valores = np.random.permutation(m)
        

    def __str__(self):
        return ' '.join(str(v) for v in self.valores) 
    
    def __repr__(self):
        return self.__str__()

class FPA(): 
    def __init__(self, num_polens = 60, taxa_corte = 0.4, p = 0.8, _lambda = 1.4, maxIt = 1000):
        self.n, self.m = None, None                         #n = dimensao de cada vetor solucao (numero de tarefas) e m = numero de maquinas
        self.numero_polens = num_polens
        self.tempos    = self.get_tempos()
        self.pop = [Polen(self.n) for _ in range(self.numero_polens)]     #POPULACAO = [X1, X2, ..., Xnumero_polens]
        self.p  = p
        self._lambda = _lambda
        self.maxIt = maxIt
        self.gestrela = None
        self.taxa_corte = taxa_corte                        #% de solucoes que serao preservadas no crossover
        self.melhor_fitness = None

    def get_tempos(self): 
        with open(sys.argv[1], 'r') as f:
            N, M = map(int, f.readline().split())

            matriz = []
            for _ in range(N):
                linha = list(map(float, f.readline().split()))
                matriz.append(linha)

        self.n = N 
        self.m = M 
        return matriz

    def fitness(self, x):
        # x e uma instancia de Polen ou um array diretamente
        # if isinstance(x, Polen):
        #     sequencia = list(map(int, x.valores % self.n))  #garante indices validos
        # else:
        #     sequencia = list(map(int, x % self.n))          #no caso de array

        # # Remove duplicatas e completa com tarefas faltantes (para garantir permutação)
        # sequencia = list(dict.fromkeys(sequencia))
        # faltantes = [i for i in range(self.n) if i not in sequencia]
        # sequencia += faltantes
        # return calcular_makespan(sequencia, self.tempos)
        return calcular_makespan(x.valores, self.tempos, self.n, self.m)
    
    def ox_crossover(self, p1, p2):
        n_preservados = int(self.n*self.taxa_corte)
        len_ = len(p1)
        def arrange(p, o, c2):
            for j in range(len_):
                i = (j + c2)%len_
                if p[i] not in o:
                    yield p[i]

        c1 = random.randrange(len_-n_preservados+1)
        c2 = c1+n_preservados
        o2 = list(p1[c1:c2])
        o1 = list(p2[c1:c2])
        r1 = list(arrange(p1, o1, c2))
        r2 = list(arrange(p2, o2, c2))
        p1 = np.array(r1[:c1] + o1 + r1[c1:])
        p2 = np.array(r2[:c1] + o2 + r2[c1:])
        return p1, p2

    def rodar(self): 
        tempo_inicio = time()
        results_fitness = []

        # Inicializa g*

        fitness = np.array(map(self.fitness, self.pop)) 
        melhor_idx = np.argmin(fitness)
        #melhor_idx = min(range(self.numero_polens), key=lambda i: self.fitness(self.pop[i]))        #cria um vetor [1, 2, ..., n] e ordena de acordo com o fitness correspondente no vetor populacional
        self.gestrela = Polen(self.m)
        self.gestrela.valores = self.pop[melhor_idx].valores.copy()
        self.melhor_fitness = self.fitness(self.gestrela)

        for it in range(self.maxIt): 
            for i in range(0, self.numero_polens, 2): 
                
                #Polinizacao global
                if np.random.rand() < self.p:
                    #gera a distribuicao de Levy 
                    L1 = levy_stable.rvs(self._lambda, beta=0, size=self.n)
                    L2 = levy_stable.rvs(self._lambda, beta=0, size=self.n)

                    pL1 = np.argsort(L1)
                    pL2 = np.argsort(L2)
                    
                    x_new = self.pop[i].valores[pL1]
                    x_new2 = self.pop[i+1].valores[pL2]
                
                #Polinizacao local
                else:
                    j = random.randint(0, self.numero_polens - 1)
                    k = random.randint(0, self.numero_polens - 1)

                    while j == k:
                        k = random.randint(0, self.numero_polens - 1)

                    #x_new    = xi1 + epsilon1 * (self.pop[j].valores - self.pop[k].valores)
                    x_new, x_new2 = self.ox_crossover(self.pop[j].valores, self.pop[k].valores)


                novo_polen = Polen(self.m)
                novo_polen.valores = x_new

                novo_polen2 = Polen(self.m)
                novo_polen2.valores = x_new2

                np1f = self.fitness(novo_polen)
                np2f = self.fitness(novo_polen2)

                # print(np1f, np2f)

                #inserindo os novos polens na populacao, caso tenham fitnesse melhor (elitismo) 
                if np1f < self.fitness(self.pop[i]):
                    self.pop[i].valores = x_new

                    # Atualiza g* se necessario
                    if np1f < self.melhor_fitness:
                        self.gestrela.valores = x_new.copy()
                        self.melhor_fitness = np1f
                
                if np2f < self.fitness(self.pop[i+1]):
                    self.pop[i+1].valores = x_new2

                    # Atualiza g* se necessario
                    if np2f < self.melhor_fitness:
                        self.gestrela.valores = x_new2.copy()
                        self.melhor_fitness = np2f

            results_fitness.append(self.melhor_fitness)
    
        self.historico_fitness = results_fitness
        tempo_fim = time()
        self.tempo_exec = tempo_fim - tempo_inicio

            
    def __str__(self):
        return f'Melhor Makespan: {self.melhor_fitness}\nMelhor Sequencia: {self.gestrela}\nTempo de Execucao (segundos): {self.tempo_exec:.4f}\n' #+ str(self.pop)
        


# for p in [0.1, 0.2, 0.5, 0.8, 0.9]: 
#     print(f'p={p}')
#     fpa = FPA()
#     fpa.rodar()
#     plt.figure(figsize=(10,8))
#     plt.scatter(range(fpa.maxIt), fpa.historico_fitness)
#     plt.savefig(f'results_p-{str(p)}.png', dpi = 400)
#     print(fpa)
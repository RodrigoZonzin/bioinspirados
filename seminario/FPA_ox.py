import numpy as np 
import random
import pandas as pd
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys
from time import time


def calcular_makespan(sequencia, tempos):
    N = len(sequencia)
    M = len(tempos[0])
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
    def __init__(self, m, valores=None):
        self.m = m
        if valores is not None:
            self.valores = np.array(valores)
        else:
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
        self.pop = [Polen(self.m) for _ in range(self.numero_polens)]     #POPULACAO = [X1, X2, ..., Xnumero_polens]
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
        if isinstance(x, Polen):
            sequencia = list(map(int, x.valores % self.n))  #garante indices validos
        else:
            sequencia = list(map(int, x % self.n))          #no caso de array

        # Remove duplicatas e completa com tarefas faltantes (para garantir permutação)
        sequencia = list(dict.fromkeys(sequencia))
        faltantes = [i for i in range(self.n) if i not in sequencia]
        sequencia += faltantes
        return calcular_makespan(sequencia, self.tempos)
    
    def ox_crossover(self, pai1, pai2):
        n_preservados = int(self.m*self.taxa_corte)
        tamanho = len(pai1)

        # Escolhe aleatoriamente a posicao de inicio para os genes preservados
        #print('m, taxa, tamanho:', self.m, self.taxa_corte, n_preservados)
        inicio = np.random.randint(0, tamanho - n_preservados + 1)
        fim = inicio + n_preservados

        # Cria filhos com None (ou -1 se preferir) nos espaços a serem preenchidos
        filho1 = [None] * tamanho
        filho2 = [None] * tamanho

        # Copia o segmento preservado dos pais
        filho1[inicio:fim] = pai1[inicio:fim]
        filho2[inicio:fim] = pai2[inicio:fim]

        # funcao auxiliar para preencher os filhos com os elementos do outro pai
        def preencher(filho, outro_pai, inicio, fim):
            idx = fim % tamanho
            for gene in outro_pai:
                if gene not in filho:
                    filho[idx] = gene
                    idx = (idx + 1) % tamanho
            return np.array(filho)

        filho1 = preencher(filho1, pai2, inicio, fim)
        filho2 = preencher(filho2, pai1, inicio, fim)

        return filho1, filho2

    def rodar(self): 
        tempo_inicio = time()
        results_fitness = []

        # Inicializa g*
        melhor_idx = min(range(self.numero_polens), key=lambda i: self.fitness(self.pop[i]))        #cria um vetor [1, 2, ..., n] e ordena de acordo com o fitness correspondente no vetor populacional
        self.gestrela = Polen(self.m)
        self.gestrela.valores = self.pop[melhor_idx].valores.copy()
        self.melhor_fitness = self.fitness(self.gestrela)

        for it in range(self.maxIt): 
            for i in range(0, self.numero_polens, 2): 
                xi1 = self.pop[i].valores.copy()            #x_{i} 
                xi2 = self.pop[i+1].valores.copy()          #x_{i+1}

                #Polinizacao global
                if np.random.rand() < self.p:
                    #gera a distribuicao de Levy 
                    L1 = levy_stable.rvs(self._lambda, beta=0, size=self.m)
                    L2 = levy_stable.rvs(self._lambda, beta=0, size=self.m)
                    
                    perturb = list(zip(xi1, L1))
                    perturb.sort(key=lambda x: x[1])
                    x_new = np.array([x[0] for x in perturb], dtype=int)                    

                    perturb = list(zip(xi2, L2))
                    perturb.sort(key=lambda x: x[1])
                    x_new2 = np.array([x[0] for x in perturb], dtype=int)                    

                
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

                #inserindo os novos polens na populacao, caso tenham fitnesse melhor (elitismo) 
                if self.fitness(novo_polen) < self.fitness(self.pop[i].valores):
                    self.pop[i].valores = x_new

                    # Atualiza g* se necessario
                    if self.fitness(novo_polen) < self.melhor_fitness:
                        self.gestrela.valores = x_new.copy()
                        self.melhor_fitness = self.fitness(novo_polen)
                
                if self.fitness(novo_polen2) < self.fitness(self.pop[i+1].valores):
                    self.pop[i+1].valores = x_new2

                    # Atualiza g* se necessario
                    if self.fitness(novo_polen2) < self.melhor_fitness:
                        self.gestrela.valores = x_new2.copy()
                        self.melhor_fitness = self.fitness(novo_polen2)

            results_fitness.append(self.melhor_fitness)
    
        self.historico_fitness = results_fitness
        tempo_fim = time()
        self.tempo_exec = tempo_fim - tempo_inicio

            
    def __str__(self):
        return f'Melhor Makespan: {self.melhor_fitness}\nMelhor Sequencia: {self.gestrela}\nTempo de Execucao (segundos): {self.tempo_exec:.4f}\n' #+ str(self.pop)
        

"""
for p in [0.8]: 
    print(f'p={p}')
    fpa = FPA()
    fpa.rodar()
    plt.figure(figsize=(10,8))
    plt.scatter(range(fpa.maxIt), fpa.historico_fitness)
    plt.savefig(f'results_p-{str(p)}.png', dpi = 400)
    print(fpa)
"""

fpa = FPA(num_polens=200, taxa_corte = 0.2, p = 0.9, _lambda = 1.5, maxIt=5000)
fpa.rodar()
plt.figure(figsize=(10,8))
plt.scatter(range(fpa.maxIt), fpa.historico_fitness)
plt.savefig(f'results_p.png', dpi = 400)
print(fpa)
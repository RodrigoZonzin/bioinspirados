import numpy as np 
import random
import pandas as pd
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys


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
    def __init__(self, n, m = 10, taxa_corte = 0.2):
        self.N, self.M = None, None
        self.intervalo = (-4, 4)
        self.tempos     = self.get_tempos()
        self.n = n                                      #dimensao de cada vetor solucao
        self.m = m                                      #numero de flores na populacao
        self.pop = [Polen(self.m) for _ in range(n)]    #POPULACAO = [X1, X2, ..., Xn]
        self.L  = None 
        self.p  = 0.8
        self._lambda = 1.5
        self.maxIt = 500
        self.gestrela = None
        self.corte = int(taxa_corte*n)
        self.melhor_fitness = None
        print('n, m:', self.n, self.m)

    def get_tempos(self): 
        with open(sys.argv[1], 'r') as f:
            N, M = map(int, f.readline().split())

            matriz = []
            for _ in range(N):
                linha = list(map(float, f.readline().split()))
                matriz.append(linha)

        self.N = N 
        self.n = N 
        self.M = M
        return matriz

    """def fitness(self, x):
        n = x.m
        x = x.valores
        part1 = -0.2 * np.sqrt(np.sum(x**2) / n)
        part2 = np.sum(np.cos(2 * np.pi * x)) / n
        return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e
    """
    def fitness(self, x):
        # x é uma instância de Polen ou um array diretamente
        if isinstance(x, Polen):
            sequencia = list(map(int, x.valores % self.N))  # garante índices válidos
        else:
            sequencia = list(map(int, x % self.N))          # no caso de array

        # Remove duplicatas e completa com tarefas faltantes (para garantir permutação)
        sequencia = list(dict.fromkeys(sequencia))
        faltantes = [i for i in range(self.N) if i not in sequencia]
        sequencia += faltantes
        return calcular_makespan(sequencia, self.tempos)
    
    def ox_crossover(self, pai1, pai2):
        n_preservados = self.corte
        tamanho = len(pai1)

        # Escolhe aleatoriamente a posição de início para os genes preservados
        inicio = np.random.randint(0, tamanho - n_preservados + 1)
        fim = inicio + n_preservados

        # Cria filhos com None (ou -1 se preferir) nos espaços a serem preenchidos
        filho1 = [None] * tamanho
        filho2 = [None] * tamanho

        # Copia o segmento preservado dos pais
        filho1[inicio:fim] = pai1[inicio:fim]
        filho2[inicio:fim] = pai2[inicio:fim]

        # Função auxiliar para preencher os filhos com os elementos do outro pai
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
        results_fitness = []

        # Inicializa g*
        melhor_idx = min(range(self.n), key=lambda i: self.fitness(self.pop[i]))
        self.gestrela = Polen(self.m)
        self.gestrela.valores = self.pop[melhor_idx].valores.copy()
        self.melhor_fitness = self.fitness(self.gestrela)

        for it in range(self.maxIt): 
            for i in range(0, self.n, 2): 
                xi1 = self.pop[i].valores.copy()            #x_{i} 
                xi2 = self.pop[i+1].valores.copy()          #x_{i+1}

                #Polinizacao global
                if np.random.rand() < self.p:
                    #gera a distribuicao de Levy 
                    L1 = levy_stable.rvs(self._lambda, beta=0, size=self.m)
                    L2 = levy_stable.rvs(self._lambda, beta=0, size=self.m)
                    x_new   = xi1 + L1*(self.gestrela.valores - xi1)
                    x_new2  = xi2 + L2*(self.gestrela.valores - xi2)
                
                #Polinizacao local
                else:
                    j = random.randint(0, self.n - 1)
                    k = random.randint(0, self.n - 1)

                    while j == k:
                        k = random.randint(0, self.n - 1)

                    epsilon1 = np.random.uniform(size=self.m)
                    epsilon2 = np.random.uniform(size=self.m)

                    #x_new    = xi1 + epsilon1 * (self.pop[j].valores - self.pop[k].valores)
                    x_new, x_new2 = self.ox_crossover(self.pop[j].valores, self.pop[k].valores)

                #assegura que xmin <= x <= xmax
                #x_new   = np.clip(x_new, self.intervalo[0], self.intervalo[1])
                #x_new2  = np.clip(x_new2, self.intervalo[0], self.intervalo[1])

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

            
    def __str__(self):
        return f'Numero de polens: {self.n}\nVetor gbest:{self.gestrela}\nMelhor Fitness: {self.melhor_fitness}\n'+ str(self.pop)
        



fpa = FPA(10, m  = 3)
fpa.rodar()
plt.scatter(range(fpa.maxIt), fpa.historico_fitness)
plt.savefig('results.png', dpi = 400)
print(fpa)
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    #def __

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, intervalo=(-2, 2)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo

    def _gerar_populacao(self):
        global pmin, pmax
        return [Gene([random.randint(pmin, pmax) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)


    #Funcao de Ackley 
    def fitness(self, gene):
        x = gene.alelos
        n = len(x)

        sum1 = sum([xi**2 for xi in x])
        sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
        parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
        parte2 = -math.exp(sum2 / n)
        return parte1 + parte2 + 20 + math.e

    #Torneio 
    def selecao(self):
        competidores = random.sample(self.pop, 2)
        return min(competidores, key=self.fitness)  

    #Roleta
    def roleta(self): 
        #obter o fitness invertido (==> min)
        valores_fitness = np.array([self.fitness(gene) for gene in self.pop])
        
        #normalizar o vetor para obter a proporcao de cada secao da roleta
        valores_fitness = valores_fitness / np.sum(valores_fitness)
        
        #sortear um numero e percorrer o vetor acumulado (usar choice)
        r = np.random.random()
        return np.random.choice(a = valores_fitness, p=r)

    def crossover(self, pai1, pai2):
        ponto = random.randint(1, self.gene_size - 1)
        filho1 = pai1.alelos[:ponto] + pai2.alelos[ponto:]
        filho2 = pai2.alelos[:ponto] + pai1.alelos[ponto:]
        return Gene(filho1), Gene(filho2)

    def crossoverBLXAlphaBeta(self, paiX, paiY, alpha = 0.75, beta = 0.25): 
        d = [abs( paiX.alelos[i] - paiY.alelos[i]) for i in range(self.gene_size)]

        print(f"Fitness de PaiX: {self.fitness(paiX)}")
        print(f"Fitness de PaiY: {self.fitness(paiY)}")

        us = []
        for i in range(self.gene_size):
            if self.fitness(paiX) <= self.fitness(paiY): 
                us.append((paiX.alelos[i]- alpha*d[i], paiY.alelos[i]+beta*d[i]))
            else:
                us.append((paiY.alelos[i]- beta*d[i], paiX.alelos[i]+alpha*d[i]))
        
        
        print(us, random.choice(us))

    #altera aleatoriamente os bits dos alelos a uma taxa self.taxa_mutacao
    def mutacao(self, gene):
        for i in range(len(gene.alelos)):
            if random.random() < self.taxa_mutacao:
                gene.alelos[i] = 1 - gene.alelos[i]
        return gene
    
    def nova_geracao(self):
        nova_pop = []
        while len(nova_pop) < self.pop_size:
            pai1 = self.selecao()
            pai2 = self.selecao()
            
            #novos filhos 
            filho1, filho2 = self.crossover(pai1, pai2)
            
            #submete filho1 à mutacao
            nova_pop.append(self.mutacao(filho1))

            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(filho2))

        self.pop = nova_pop

    def nova_geracao_elitismo(self):
        nova_pop = []

        #add melhor gene na populacao
        nova_pop.append(self.melhor_individuo())  

        while len(nova_pop) < self.pop_size:
            pai1 = self.selecao()
            pai2 = self.selecao()

            #novos filhos
            filho1, filho2 = self.crossover(pai1, pai2)

            #submete filho1 à mutacao
            nova_pop.append(self.mutacao(filho1))

            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(filho2))
        self.pop = nova_pop

    def melhor_individuo(self):
        return min(self.pop, key=self.fitness)


if __name__ == "__main__":
    pmin = -2
    pmax = 5
    random.seed(42) 

    ag = AG(pop_size=4, gene_size=3, taxa_mutacao=0.01)  
    geracoes = 2

    ag.print_pop()
    print(f"X= {ag.pop[0]}, Y={ag.pop[2]}")
    ag.crossoverBLXAlphaBeta(ag.pop[0], ag.pop[2])

    """for i in range(geracoes):
        print(f"Geração {i}")
        ag.print_pop()
        
        #melhor gene
        melhor = ag.melhor_individuo()
        print(f"Melhor: {melhor}")
        print(f"x = {ag.bin_to_real(melhor)}")

        #fitness do melhor gene
        print(f"Fitness: {ag.fitness(melhor)}\n")
        ag.nova_geracao_elitismo()
        
    """



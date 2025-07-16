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

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, intervalo=(-2, 2)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo


    #ver esse pmin, pmax depois
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

    #Roleta
    def roleta(self): 
        #obter o fitness invertido (==> min)
        valores_fitness = np.array([1/self.fitness(gene) for gene in self.pop])
        
        #normalizar o vetor para obter a proporcao de cada secao da roleta
        valores_fitness = valores_fitness / np.sum(valores_fitness)

        return self.pop[np.random.choice(a = range(self.pop_size), replace = False, p=valores_fitness)]

    def crossoverBLXAlphaBeta(self, paiX, paiY, alpha = 0.75, beta = 0.25): 
        d = [abs(paiX.alelos[i] - paiY.alelos[i]) for i in range(self.gene_size)]

        print(f"Fitness de PaiX: {self.fitness(paiX)}")
        print(f"Fitness de PaiY: {self.fitness(paiY)}")

        us = []
        for i in range(self.gene_size):
            #cuidar p nao extrapolar (min, max)
            if self.fitness(paiX) <= self.fitness(paiY): 
                x1 = paiX.alelos[i]- alpha*d[i] if paiX.alelos[i]- alpha*d[i] >= -2 else -2
                x2 = paiY.alelos[i]+beta*d[i]   if paiY.alelos[i]+beta*d[i]    <=2 else 2
                us.append((x1, x2))
            else:
                x1 = paiY.alelos[i]- beta*d[i] if paiY.alelos[i]- beta*d[i] >= -2 else -2
                x2 = paiX.alelos[i]+alpha*d[i] if paiX.alelos[i]+alpha*d[i] <= 2 else 2
                us.append((x1,x2))
        
        
        print(us, f"Intervalo sorteado: {random.choice(us)}")

    #altera aleatoriamente os bits dos alelos a uma taxa self.taxa_mutacao
    #conferir pmax
    def mutacao(self, gene, pmax = 10):
        for i in range(len(gene.alelos)):
            if random.random() < self.taxa_mutacao:
                gene.alelos[i] = random.randint(pmax)
        return gene
    
    def nova_geracao(self):
        nova_pop = []
        while len(nova_pop) < self.pop_size:
            #verificar se pai1 != 
            pai1 = self.roleta()
            pai2 = pai1

            while pai1 == pai2:
                pai2 = self.roleta()

            
            #novos filhos 
            filho1, filho2 = self.crossoverBLXAlphaBeta(pai1, pai2)
            print(f'f1={filho1}, f2={filho2}')

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
            pai1 = self.roleta()
            pai2 = self.roleta()

            #novos filhos
            filho1, filho2 = self.crossoverBLXAlphaBeta(pai1, pai2)

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

    for i in range(geracoes):
        print(f"Geração {i}")
        ag.print_pop()
        
        #melhor gene
        melhor = ag.melhor_individuo()
        print(f"Melhor: {melhor}")
        

        #fitness do melhor gene
        print(f"Fitness: {ag.fitness(melhor)}\n")
        ag.nova_geracao()
        



import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os


class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, intervalo=(-2, 2), minMaxIndiv = (-20, 20)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.intervalo = intervalo
        self.minMaxIndiv = minMaxIndiv
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        

    def _gerar_populacao(self):
        vmin, vmax = self.minMaxIndiv
        return [Gene([random.uniform(vmin, vmax) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)

    def fitness(self, gene):
        x = gene.alelos
        n = len(x)
        sum1 = sum([xi ** 2 for xi in x])
        sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
        parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
        parte2 = -math.exp(sum2 / n)
        return parte1 + parte2 + 20 + math.e

    def roleta(self):
        valores_fitness = np.array([1 / self.fitness(gene) for gene in self.pop])
        valores_fitness = valores_fitness / np.sum(valores_fitness)
        return self.pop[np.random.choice(a=range(self.pop_size), replace=False, p=valores_fitness)]

    def crossoverBLXAlphaBeta(self, paiX, paiY, alpha=0.75, beta=0.25):
        filhos = []

        d = [abs(paiX.alelos[i] - paiY.alelos[i]) for i in range(self.gene_size)]

        for _ in range(2):  
            alelos_filho = []
            for i in range(self.gene_size):
                if self.fitness(paiX) <= self.fitness(paiY):
                    x1 = max(paiX.alelos[i] - alpha * d[i], self.intervalo[0])
                    x2 = min(paiY.alelos[i] + beta * d[i], self.intervalo[1])
                else:
                    x1 = max(paiY.alelos[i] - beta * d[i], self.intervalo[0])
                    x2 = min(paiX.alelos[i] + alpha * d[i], self.intervalo[1])
                alelos_filho.append(random.uniform(x1, x2))
            filhos.append(Gene(alelos_filho))

        return filhos[0], filhos[1]

    def mutacao(self, gene):
        pmin, pmax = self.intervalo
        for i in range(len(gene.alelos)):
            if random.random() < self.taxa_mutacao:
                gene.alelos[i] = random.uniform(pmin, pmax)
        return gene

    def mutacao_suave(self, gene):
        for i in range(len(gene.alelos)):
            if random.random() < self.taxa_mutacao:
                perturbacao = random.gauss(0, 0.1)  
                gene.alelos[i] += perturbacao

                gene.alelos[i] = max(self.intervalo[0], min(self.intervalo[1], gene.alelos[i]))
        return gene


    def nova_geracao(self):
        nova_pop = []
        while len(nova_pop) < self.pop_size:
            pai1 = self.roleta()
            pai2 = self.roleta()
            while pai1 == pai2:
                pai2 = self.roleta()

            filho1, filho2 = self.crossoverBLXAlphaBeta(pai1, pai2)

            nova_pop.append(self.mutacao_suave(filho1))

            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao_suave(filho2))

        self.pop = nova_pop

    def nova_geracao_elitismo(self):
        nova_pop = [self.melhor_individuo()]

        while len(nova_pop) < self.pop_size:
            pai1 = self.roleta()
            pai2 = self.roleta()

            filho1, filho2 = self.crossoverBLXAlphaBeta(pai1, pai2)

            nova_pop.append(self.mutacao(filho1))

            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(filho2))

        self.pop = nova_pop

    def melhor_individuo(self):
        return min(self.pop, key=self.fitness)


if __name__ == "__main__":
    #random.seed(42)
    os.makedirs('resultsMutacao', exist_ok=True)
    
    n_execucoes = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
    results_execucoes = []
    for exec in n_execucoes:
        with open(f'resultsMutacao/logAG_{exec}.txt', 'w') as f:
            sys.stdout = f
            fitness_result = []
            
            ag = AG(pop_size=20, gene_size=5, taxa_mutacao=exec, intervalo=(-2, 2))
            
            geracoes = 60
            for i in range(geracoes):
                print(f"Geracao {i}")
                ag.print_pop()

                melhor = ag.melhor_individuo()
                print(f"\nMelhor individuo: {melhor}")
                print(f"Fitness: {ag.fitness(melhor)}\n")
                
                fitness_result.append(ag.fitness(melhor))
                ag.nova_geracao_elitismo()

            results_execucoes.append(fitness_result)


            sys.stdout = sys.__stdout__
        

    
    plt.figure(figsize=(15, 10))
    for i,exec in enumerate(n_execucoes):
        plt.plot(range(geracoes), results_execucoes[i], label = f'Taxa Mutacao. {exec}')

    plt.legend()
    plt.xlabel("Gerações")
    plt.ylabel("Fitness")
    plt.savefig('resultsMutacao/fitness_mutacao.png')
        
    #print(results_execucoes)

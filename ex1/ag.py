import numpy as np
import random
import math 

class Gene(): 
    def __init__(self, alelos: tuple()):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    """
    def init_random(self, size): 
        for i in range(size): 
            self.alelos[i] = 
    """
    
    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim

class AG(): 
    def __init__(self, pop_size = 10, gene_size = 8, taxa_mutacao= 0.01, intervalo =(-5, 5)): 
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pop  = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo
        self.n_var = gene_size // 10
 
    def _gerar_populacao(self):
        return [Gene([random.randint(0, 1) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        [print(i) for i in self.pop]

    def bin_to_int(self, gene):
        blocos = [gene.alelos[i:i + 10] for i in range(0, len(gene.alelos), 10)]
        reais = []
        for bloco in blocos:
            inteiro = int(''.join(map(str, bloco)), 2)
            x = self.intervalo[0] + (inteiro / (2**10 - 1)) * (self.intervalo[1] - self.intervalo[0])
            reais.append(x)
        return reais
    
    def dim_pop(self):
        return len(self.pop)

    def fitness(self, gene):
        x = self.bin_to_int(gene)
        n = len(x)

        sum1 = sum([xi**2 for xi in x])
        sum2 = sum([np.cos(2*np.pi*xi) for xi in x])

        parte1 = -20*np.exp(0.2*np.sqrt(sum1/n))
        parte2 = -np.exp(parte2/n)
        return parte1 + parte2 +20 + np.exp(1)

    def selecao(self):
        competidores = random.sample(self.pop, 2)
        return min(competidores, key=self.fitness)

    def crossover(self, pai1, pai2):
        ponto = random.randint(1, self.gene_size - 1)
        filho1 = pai1.alelos[:ponto] + pai2.alelos[ponto:]
        filho2 = pai2.alelos[:ponto] + pai1.alelos[ponto:]
        return Gene(filho1), Gene(filho2)

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
            filho1, filho2 = self.crossover(pai1, pai2)
            nova_pop.append(self.mutacao(filho1))
            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(filho2))
        self.pop = nova_pop

    def melhor_individuo(self):
        return max(self.pop, key=self.fitness)

    def f(x):
        return -20 * np.exp(-0.2 * np.sqrt((1/n) * sum(x**2))) - np.exp((1/n) * sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)


ag = AG(pop_size = 10, gene_size = 40, taxa_mutacao = 0.01)
geracoes = 20

for g in range(geracoes):
    print(f"Geracao {g}")
    ag.print_pop()
    melhor = ag.melhor_individuo()
    print(f"Melhor: {melhor}, Fitness: {ag.fitness(melhor)}\n")
    ag.nova_geracao()


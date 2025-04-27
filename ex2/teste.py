import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim

class AG:
    def __init__(self, pop_size=10, gene_size=6, taxa_mutacao=0.1, intervalo=(-2, 2)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo
        self.pop = self._gerar_populacao()

    def _gerar_populacao(self):
        return [Gene([random.uniform(self.intervalo[0], self.intervalo[1]) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)

    # Função de Ackley
    def fitness(self, gene):
        x = gene.alelos
        n = len(x)

        sum1 = sum([xi**2 for xi in x])
        sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
        parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
        parte2 = -math.exp(sum2 / n)
        return parte1 + parte2 + 20 + math.e

    # Seleção por roleta
    def roleta(self):
        valores_fitness = np.array([1 / (1 + self.fitness(gene)) for gene in self.pop])
        valores_fitness = valores_fitness / np.sum(valores_fitness)
        return self.pop[np.random.choice(a=range(self.pop_size), replace=True, p=valores_fitness)]

    # Crossover BLX-Alpha-Beta
    def crossoverBLXAlphaBeta(self, paiX, paiY, alpha=0.75, beta=0.25):
        filho1 = []
        filho2 = []
        for i in range(self.gene_size):
            d = abs(paiX.alelos[i] - paiY.alelos[i])
            x_min = min(paiX.alelos[i], paiY.alelos[i]) - alpha * d
            x_max = max(paiX.alelos[i], paiY.alelos[i]) + beta * d

            # Garantir limites
            x_min = max(self.intervalo[0], x_min)
            x_max = min(self.intervalo[1], x_max)

            filho1.append(random.uniform(x_min, x_max))
            filho2.append(random.uniform(x_min, x_max))

        return Gene(filho1), Gene(filho2)

    # Mutação suave (pequena perturbação)
    def mutacao(self, gene):
        for i in range(len(gene.alelos)):
            if random.random() < self.taxa_mutacao:
                perturbacao = random.gauss(0, 0.1)
                gene.alelos[i] += perturbacao
                gene.alelos[i] = max(self.intervalo[0], min(self.intervalo[1], gene.alelos[i]))
        return gene

    def nova_geracao_elitismo(self):
        nova_pop = []
        nova_pop.append(self.melhor_individuo())

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
    random.seed(42)
    
    ag = AG(pop_size=50, gene_size=6, taxa_mutacao=0.1, intervalo=(-2, 2))
    geracoes = 100

    historico = []

    for i in range(geracoes):
        melhor = ag.melhor_individuo()
        historico.append(ag.fitness(melhor))
        print(f"Geração {i}: Melhor fitness = {ag.fitness(melhor)}")

        ag.nova_geracao_elitismo()

    print("\nFinal:")
    melhor = ag.melhor_individuo()
    print(f"Melhor indivíduo: {melhor.alelos}")
    print(f"Fitness: {ag.fitness(melhor)}")

    # Plotar a convergência
    plt.plot(historico)
    plt.xlabel('Geração')
    plt.ylabel('Melhor Fitness')
    plt.title('Convergência do AG na função de Ackley')
    plt.grid()
    plt.show()

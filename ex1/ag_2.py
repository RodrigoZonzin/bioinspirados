import numpy as np
import random
import math

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return ''.join(map(str, self.alelos))

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, intervalo=(-5, 5)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo
        self.n_var = gene_size // 10  # usando 10 bits por variável

    def _gerar_populacao(self):
        return [Gene([random.randint(0, 1) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)

    def bin_to_real(self, gene):
        # Divide o gene em blocos de 10 bits e mapeia cada bloco para o intervalo real
        blocos = [gene.alelos[i:i + 10] for i in range(0, len(gene.alelos), 10)]
        reais = []
        for bloco in blocos:
            inteiro = int(''.join(map(str, bloco)), 2)
            x = self.intervalo[0] + (inteiro / (2**10 - 1)) * (self.intervalo[1] - self.intervalo[0])
            reais.append(x)
        return reais

    def fitness(self, gene):
        x = self.bin_to_real(gene)
        n = len(x)
        sum1 = sum([xi**2 for xi in x])
        sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
        parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
        parte2 = -math.exp(sum2 / n)
        return parte1 + parte2 + 20 + math.e

    def selecao(self):
        competidores = random.sample(self.pop, 2)
        return min(competidores, key=self.fitness)  # menor fitness é melhor

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
        return min(self.pop, key=self.fitness)


# Execução
if __name__ == "__main__":
    ag = AG(pop_size=70, gene_size=40, taxa_mutacao=0.02)  # 4 variáveis (40 bits)
    geracoes = 50

    for g in range(geracoes):
        print(f"Geração {g}")
        ag.print_pop()
        melhor = ag.melhor_individuo()
        print(f"Melhor: {melhor}")
        print(f"x = {ag.bin_to_real(melhor)}")
        print(f"Fitness (Ackley): {ag.fitness(melhor)}\n")
        ag.nova_geracao()


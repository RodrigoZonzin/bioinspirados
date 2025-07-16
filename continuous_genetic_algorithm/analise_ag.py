import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim

class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, taxa_cruzamento=1.0, intervalo=(-2, 2), minMaxIndiv=(-20, 20)):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.intervalo = intervalo
        self.minMaxIndiv = minMaxIndiv
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento  # Novo parâmetro para controle de cruzamento

    def _gerar_populacao(self):
        vmin, vmax = self.minMaxIndiv
        return [Gene([random.uniform(vmin, vmax) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

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

    def nova_geracao_elitismo(self):
        nova_pop = [self.melhor_individuo()]
        while len(nova_pop) < self.pop_size:
            pai1 = self.roleta()
            pai2 = self.roleta()
            if random.random() < self.taxa_cruzamento:
                filho1, filho2 = self.crossoverBLXAlphaBeta(pai1, pai2)
            else:
                filho1, filho2 = pai1, pai2

            nova_pop.append(self.mutacao(filho1))
            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(filho2))
        self.pop = nova_pop

    def melhor_individuo(self):
        return min(self.pop, key=self.fitness)

# -------------------------------------
# Parte para rodar o experimento fatorial
# -------------------------------------

# Configurações dos testes
taxas_mutacao = [0.01, 0.05, 0.10]
taxas_cruzamento = [0.6, 0.8, 1.0]
populacoes = [25, 50, 100]
geracoes_lista = [25, 50, 100]

# Resultados
resultados = []

# Experimento
for mutacao in taxas_mutacao:
    for cruzamento in taxas_cruzamento:
        for pop in populacoes:
            for geracoes in geracoes_lista:
                ag = AG(pop_size=pop, gene_size=3, taxa_mutacao=mutacao, taxa_cruzamento=cruzamento, intervalo=(-2, 2))
                for _ in range(geracoes):
                    ag.nova_geracao_elitismo()
                melhor_fitness = ag.fitness(ag.melhor_individuo())
                resultados.append({
                    'Melhor Fitness': melhor_fitness,
                    'Mutação': mutacao,
                    'Cruzamento': cruzamento,
                    'População': pop,
                    'Gerações': geracoes
                })
                print(f"Feito: Mutação={mutacao}, Cruzamento={cruzamento}, População={pop}, Gerações={geracoes}, Fitness={melhor_fitness}")

# Criar DataFrame
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

# Opcional: salvar em CSV
df_resultados.to_csv('resultados_ag.csv', index=False)

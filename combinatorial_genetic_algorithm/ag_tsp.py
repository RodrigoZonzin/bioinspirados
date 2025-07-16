import numpy as np
import random
import matplotlib.pyplot as plt
import os


def read_data(name: str):
    return np.loadtxt(name)

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos  
        self.alelos_dim = len(alelos)

    def __str__(self):
        return ''.join(map(str, self.alelos))

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, matrix, pop_size=50, taxa_mutacao=0.01):
        self.dist_matrix    = matrix
        self.matrix_shape   = matrix.shape
        self.pop_size       = pop_size
        self.gene_size      = self.matrix_shape[0] #como a matrix será sempre NxN, cada gene terá N elementos  
        self.taxa_mutacao   = taxa_mutacao
        self.pop            = self._gerar_populacao()

    def _gerar_populacao(self):
        return [Gene([random.randint(self.gene_size, self.gene_size) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)
        
    def fitness(self, gene):
        f = np.sum([(self.matrix[gene.alelos[i], gene.alelos[i+1]]+ self.matrix[self.gene_size, 1])  for i in range(self.gene_size) ] )
        return f


    def selecao(self):
        competidores = random.sample(self.pop, 2)
        return max(competidores, key=self.fitness)  #max

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

    def nova_geracao_elitismo(self):
        nova_pop = [self.melhor_individuo()]  #mantem o melhor
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

    def peso_mochila(self, gene): 
        return np.sum([self.pesos[i] for i in range(gene.alelos_dim) if gene.alelos[i] == 1])

if __name__ == "__main__":
    c, p, s, w = get_values()
    nexec = 20

    fig = plt.figure(figsize=(10, 11))

    for exec in range(nexec):
        ag = AG(p, w, c, pop_size=50, taxa_mutacao=0.01)
        geracoes = 100
        fitness_result = []
        geracoes_result = []

        for i in range(geracoes):
            melhor = ag.melhor_individuo()
            print(f"Geração {i}: Melhor solução = {melhor}, Fitness = {ag.fitness(melhor)}, Peso Utilizado = {ag.peso_mochila(melhor)}")
            fitness_result.append(ag.fitness(melhor))
            geracoes_result.append(i)
            ag.nova_geracao_elitismo()

        plt.plot(geracoes_result, fitness_result, label = "Ger."+str(exec+1))

    plt.xlabel("Gerações")
    plt.ylabel("Valor Total da Mochila")
    plt.legend()
    plt.savefig('resultsss.png')
    #plt.show()

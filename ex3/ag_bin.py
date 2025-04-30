import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

#maximizacao, nao minimizacao

class Gene:
    def __init__(self, alelos, pesos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)
        self.pesos = pesos 

    def __str__(self):
        return ''.join(map(str, self.alelos))

    def shape(self):
        return self.alelos_dim


class AG:
    def __init__(self, pop_size=10, gene_size=40, taxa_mutacao=0.01, intervalo=(-2, 2), bits_var = 5):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pop = self._gerar_populacao()
        self.taxa_mutacao = taxa_mutacao
        self.intervalo = intervalo

    def _gerar_populacao(self):
        return [Gene([random.randint(0, 1) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)

    #Funcao  
    def fitness(self, gene, c = 10):
        somatorio_v = np.sum(gene.alelos)
        somatorio_p = np.sum(gene.pesos)

        return somatorio_v - (somatorio_v*(somatorio_p - c))


    #Torneio 
    def selecao(self):
        competidores = random.sample(self.pop, 2)
        return min(competidores, key=self.fitness)  

    #000|01 x 100|11  ==> 00011, 10001
    def crossover(self, pai1, pai2):
        ponto = random.randint(1, self.gene_size - 1)
        filho1 = pai1.alelos[:ponto] + pai2.alelos[ponto:]
        filho2 = pai2.alelos[:ponto] + pai1.alelos[ponto:]
        return Gene(filho1), Gene(filho2)

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
    ag = AG(pop_size=50, gene_size=20, taxa_mutacao=0.01)  
    geracoes = 100

    

    """for i in range(geracoes):
        ag.print_pop()
        
        #melhor gene
        melhor = ag.melhor_individuo()
        o_file.write(f"Melhor: {melhor}")
        o_file.write(f"x = {ag.bin_to_real(melhor)}")

        #fitness do melhor gene
        o_file.write(f"Fitness: {ag.fitness(melhor)}\n")
        fitness_result.append(ag.fitness(melhor))

        geracoes_result.append(i)
        ag.nova_geracao_elitismo()
        
    cmap = mpl.colormaps['plasma'].colors 

    plt.scatter(geracoes_result, fitness_result, color = cmap[0])
    plt.xlabel("Gerações")
    plt.ylabel(r"f(x)")

    plt.savefig(f'fitness_{sys.argv[1]}.png')"""




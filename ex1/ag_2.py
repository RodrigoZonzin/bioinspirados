import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

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
        self.n_var = gene_size // bits_var  #cada variavel será representada por gene_size//10 bits. 
                                            #ex para gene_size = 15, bits_por_variavel:
                                            #100001111111000 ==> x1=10000 x2=11111 x3=11000 

    def _gerar_populacao(self):
        return [Gene([random.randint(0, 1) for _ in range(self.gene_size)]) for _ in range(self.pop_size)]

    def print_pop(self):
        for gene in self.pop:
            print(gene)

    #dado um gene a1a2a3...an, representando m variaveis, retorna-se um vetor
    #com dimensao m para as respectivas representacoes reais 
    #ex com n = 15 e m =3
    #100001111111000 ==> x1=10000 x2=11111 x3=11000 
    def bin_to_real(self, gene):
        blocos = [gene.alelos[i:i + 10] for i in range(0, len(gene.alelos), 10)]
        reais = []
        for bloco in blocos:
            inteiro = int(''.join(map(str, bloco)), 2)
            x = self.intervalo[0] + (inteiro / (2**10 - 1)) * (self.intervalo[1] - self.intervalo[0])
            reais.append(x)
        return reais

    #Funcao de Ackley 
    def fitness(self, gene):
        x = self.bin_to_real(gene)
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
    ag = AG(pop_size=70, gene_size=40, taxa_mutacao=0.02)  
    geracoes = 50

    fitness_result = []
    geracoes_result = []

    for i in range(geracoes):
        print(f"Geração {i}")
        ag.print_pop()
        
        #melhor gene
        melhor = ag.melhor_individuo()
        print(f"Melhor: {melhor}")
        print(f"x = {ag.bin_to_real(melhor)}")

        #fitness do melhor gene
        print(f"Fitness: {ag.fitness(melhor)}\n")
        fitness_result.append(ag.fitness(melhor))

        geracoes_result.append(i)
        ag.nova_geracao_elitismo()
        
    plt.scatter(geracoes_result, fitness_result)
    plt.xlabel("Gerações")
    plt.ylabel(r"f(x)")

    plt.savefig('fitness_elitismo.png')




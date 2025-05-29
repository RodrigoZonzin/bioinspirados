import numpy as np
import random
import matplotlib.pyplot as plt

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos

    def __str__(self):
        return str(self.alelos)

    def copy(self):
        return Gene(self.alelos.copy())


class AG:
    def __init__(self, matriz_dist, pop_size=200, taxa_mutacao=0.01, taxa_cruzamento=1.0, metodo_selecao="torneio"):
        self.matriz_dist = matriz_dist
        self.pop_size = pop_size
        self.gene_size = matriz_dist.shape[0]
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento
        self.metodo_selecao = metodo_selecao
        self.pop = self._gerar_populacao()

    def _gerar_populacao(self):
        return [Gene(random.sample(range(self.gene_size), self.gene_size)) for _ in range(self.pop_size)]

    def fitness(self, gene):
        caminho = gene.alelos
        dist = sum(self.matriz_dist[caminho[i]][caminho[i + 1]] for i in range(len(caminho) - 1))
        dist += self.matriz_dist[caminho[-1]][caminho[0]]
        return dist

    def selecao(self):
        if self.metodo_selecao == "torneio":
            candidatos = random.sample(self.pop, 2)
            melhor = min(candidatos, key=self.fitness)
            if random.random() > 0.9:
                melhor = candidatos[0] if melhor != candidatos[0] else candidatos[1]
            return melhor
        elif self.metodo_selecao == "roleta":
            inv_fitness = [1 / self.fitness(ind) for ind in self.pop]
            soma = sum(inv_fitness)
            probs = [f / soma for f in inv_fitness]
            return np.random.choice(self.pop, p=probs)
        else:
            raise ValueError("Método de seleção inválido.")

    def crossover(self, pai1, pai2):
        if random.random() > self.taxa_cruzamento:
            return pai1.copy(), pai2.copy()

        start, end = sorted(random.sample(range(self.gene_size), 2))

        def cria_filho(p1, p2):
            meio = p1.alelos[start:end]
            restante = [g for g in p2.alelos if g not in meio]
            return Gene(restante[:start] + meio + restante[start:])

        return cria_filho(pai1, pai2), cria_filho(pai2, pai1)

    def mutacao(self, gene):
        if random.random() < self.taxa_mutacao:
            i, j = random.sample(range(self.gene_size), 2)
            gene.alelos[i], gene.alelos[j] = gene.alelos[j], gene.alelos[i]
        return gene

    def melhor_individuo(self):
        return min(self.pop, key=self.fitness)

    def nova_geracao_elitismo(self):
        nova_pop = [self.melhor_individuo().copy()]
        while len(nova_pop) < self.pop_size:
            pai1 = self.selecao()
            pai2 = self.selecao()
            f1, f2 = self.crossover(pai1, pai2)
            nova_pop.append(self.mutacao(f1))
            if len(nova_pop) < self.pop_size:
                nova_pop.append(self.mutacao(f2))
        self.pop = nova_pop

    def executar(self, geracoes=500):
        historico = []
        for g in range(geracoes):
            melhor = self.melhor_individuo()
            fit = self.fitness(melhor)
            print(f"Geração {g}: melhor distância = {fit}")
            historico.append(fit)
            self.nova_geracao_elitismo()
        return historico


def plotar_resultado(resultados):
    plt.plot(range(len(resultados)), resultados, marker='.', label="Melhor por geração")
    plt.title('AG para TSP')
    plt.xlabel('Gerações')
    plt.ylabel('Distância total')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    arquivo = "lau15_dist.txt"
    matriz = np.loadtxt(arquivo)
    ag = AG(matriz, pop_size=200, taxa_mutacao=0.01, taxa_cruzamento=1.0, metodo_selecao="torneio")
    resultados = ag.executar(geracoes=500)

    with open("saida_1.txt", "w") as f:
        for r in sorted(resultados, reverse=True):
            f.write(f"{r}\n")

    plotar_resultado(resultados)


if __name__ == "__main__":
    main()

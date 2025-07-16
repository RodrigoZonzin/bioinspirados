import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

cores = sns.color_palette("Set2", 2)

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos

    def __str__(self):
        return str(self.alelos)

    def copy(self):
        return Gene(self.alelos.copy())


class AG:
    def __init__(self, matriz_dist, pop_size=200, taxa_mutacao=0.01, taxa_cruzamento=1.0, metodo_selecao="torneio", metodo_crossover="pmx"):
        self.matriz_dist = matriz_dist
        self.pop_size = pop_size
        self.gene_size = matriz_dist.shape[0]
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento
        self.metodo_selecao = metodo_selecao
        self.metodo_crossover = metodo_crossover
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

        if self.metodo_crossover == "pmx":
            return self.crossover_pmx(pai1, pai2), self.crossover_pmx(pai2, pai1)
        elif self.metodo_crossover == "cx":
            return self.crossover_cx(pai1, pai2), self.crossover_cx(pai2, pai1)
        elif self.metodo_crossover == "ox":
            return self.crossover_ox(pai1, pai2), self.crossover_ox(pai2, pai1)
        else:
            raise ValueError("Método de crossover inválido.")

    def crossover_pmx(self, p1, p2):
        size = len(p1.alelos)
        start, end = sorted(random.sample(range(size), 2))
        f = [None] * size
        f[start:end] = p1.alelos[start:end]
        mapping = {p2.alelos[i]: p1.alelos[i] for i in range(start, end)}
        for i in range(size):
            if not (start <= i < end):
                val = p2.alelos[i]
                while val in f:
                    val = mapping.get(val, val)
                f[i] = val
        return Gene(f)

    def crossover_cx(self, p1, p2):
        size = len(p1.alelos)
        filho = [None] * size
        index = 0
        while filho[index] is None:
            filho[index] = p1.alelos[index]
            index = p1.alelos.index(p2.alelos[index])
        for i in range(size):
            if filho[i] is None:
                filho[i] = p2.alelos[i]
        return Gene(filho)

    def crossover_ox(self, p1, p2):
        size = len(p1.alelos)
        start, end = sorted(random.sample(range(size), 2))
        filho = [None] * size
        filho[start:end] = p1.alelos[start:end]
        remaining = [item for item in p2.alelos if item not in filho[start:end]]
        ptr = 0
        for i in range(size):
            if filho[i] is None:
                filho[i] = remaining[ptr]
                ptr += 1
        return Gene(filho)

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


def plotar_resultado(resultados, metodo, i):
    #plt.figure(figsize=(10, 8))
    plt.plot(range(len(resultados)), resultados, marker='.', label=f"Melhor por geração ({metodo})", color = cores[i])
    plt.title('AG para TSP')
    plt.xlabel('Gerações')
    plt.ylabel('Distância total')
    #plt.grid(True)
    plt.legend()
    plt.savefig(f'results_ter_{metodo}.png', dpi = 300)
    #plt.show()


def main():
    arquivo = "lau15_dist.txt"
    matriz = np.loadtxt(arquivo)
    for i, metodo in enumerate(['cx', 'ox']):
        ag = AG(matriz, pop_size=200, taxa_mutacao=0.01, taxa_cruzamento=1.0, metodo_selecao="torneio", metodo_crossover=metodo)
        resultados = ag.executar(geracoes=500)

        with open(f"saida_{metodo}.txt", "w") as f:
            for r in sorted(resultados, reverse=True):
                f.write(f"{r}\n")

        plotar_resultado(resultados, metodo, i)


if __name__ == "__main__":
    main()
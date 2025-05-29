import numpy as np
import random
import itertools
import csv
from statistics import mean, stdev
from ag_tsp2 import AG, Gene

# Novos cruzamentos
def crossover_ox(p1, p2):
    size = len(p1.alelos)
    start, end = sorted(random.sample(range(size), 2))
    filho = [None] * size
    filho[start:end] = p1.alelos[start:end]
    ptr = end
    for gene in p2.alelos:
        if gene not in filho:
            while filho[ptr % size] is not None:
                ptr += 1
            filho[ptr % size] = gene
    return Gene(filho)

def crossover_pmx(p1, p2):
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

def crossover_cx(p1, p2):
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

# Operadores disponíveis
operadores = {
    "OX": lambda a, b: (crossover_ox(a, b), crossover_ox(b, a)),
    "PMX": lambda a, b: (crossover_pmx(a, b), crossover_pmx(b, a)),
    "CX": lambda a, b: (crossover_cx(a, b), crossover_cx(b, a))
}

# Subclasse com crossover selecionável
class AGCustom(AG):
    def __init__(self, matriz_dist, operador='OX', **kwargs):
        super().__init__(matriz_dist, **kwargs)
        self.operador = operador

    def crossover(self, pai1, pai2):
        if random.random() > self.taxa_cruzamento:
            return pai1.copy(), pai2.copy()
        return operadores[self.operador](pai1, pai2)

# Teste fatorial
def rodar_experimentos():
    matriz = np.loadtxt("lau15_dist.txt")

    tamanhos = [100, 200]
    taxas_mutacao = [0.01, 0.05]
    selecoes = ["torneio", "roleta"]
    operadores_testar = ["OX", "CX"]

    resultados = []

    for pop, mut, sel, op in itertools.product(tamanhos, taxas_mutacao, selecoes, operadores_testar):
        melhores = []
        for _ in range(5):  # múltiplas execuções
            ag = AGCustom(matriz, pop_size=pop, taxa_mutacao=mut,
                          taxa_cruzamento=1.0, metodo_selecao=sel,
                          operador=op)
            historico = ag.executar(geracoes=200)
            melhores.append(historico[-1])

        resultados.append({
            "Operador": op,
            "População": pop,
            "Mutação": mut,
            "Seleção": sel,
            "Melhor": min(melhores),
            "Média": mean(melhores),
            "Desvio": stdev(melhores)
        })

    # Salvar em CSV
    with open("resultados_ag_2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

if __name__ == "__main__":
    rodar_experimentos()

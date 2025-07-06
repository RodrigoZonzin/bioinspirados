import numpy as np


def ox_crossover(pai1, pai2, n_preservados):
    tamanho = len(pai1)

    # Escolhe aleatoriamente a posição de início para os genes preservados
    inicio = np.random.randint(0, tamanho - n_preservados + 1)
    fim = inicio + n_preservados

    # Cria filhos com None (ou -1 se preferir) nos espaços a serem preenchidos
    filho1 = [None] * tamanho
    filho2 = [None] * tamanho

    # Copia o segmento preservado dos pais
    filho1[inicio:fim] = pai1[inicio:fim]
    filho2[inicio:fim] = pai2[inicio:fim]

    # Função auxiliar para preencher os filhos com os elementos do outro pai
    def preencher(filho, outro_pai, inicio, fim):
        idx = fim % tamanho
        for gene in outro_pai:
            if gene not in filho:
                filho[idx] = gene
                idx = (idx + 1) % tamanho
        return np.array(filho)

    filho1 = preencher(filho1, pai2, inicio, fim)
    filho2 = preencher(filho2, pai1, inicio, fim)

    return filho1, filho2


pai1 = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10]
pai2 = [10, 8,  1,  4,  5,  7,  2,  3,  6,  9] 

print(pai1, pai2)

f1, f2 = ox_crossover(pai1, pai2, 5)
print(f1, f2)

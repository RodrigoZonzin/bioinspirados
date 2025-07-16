import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

def carregar_dados_cidades():
    with open("instancia2.txt", 'r') as arquivo:
        linhas = arquivo.readlines()
        numero_cidades = len(linhas)
        matriz = []

        for linha in linhas:
            dados_linha = list(map(float, linha.strip().split()))
            matriz.append(dados_linha)

    return numero_cidades, np.array(matriz)

def calcular_fitness(caminho, matriz_distancias):
    distancia_total = sum(matriz_distancias[caminho[i], caminho[i + 1]] for i in range(len(caminho) - 1))
    distancia_total += matriz_distancias[caminho[-1], caminho[0]]
    return 1 / distancia_total

def criar_populacao(tamanho_pop, numero_cidades):
    return [np.random.permutation(numero_cidades) for _ in range(tamanho_pop)]

def expandir_populacao(populacao_atual, incremento, numero_cidades):
    nova_populacao = copy.deepcopy(populacao_atual)
    for _ in range(incremento):
        nova_populacao.append(np.random.permutation(numero_cidades))
    return nova_populacao

def realizar_clonagem(individuo, fitness, taxa_clonagem, tamanho_populacao):
    num_clones = int(round(taxa_clonagem * tamanho_populacao / (fitness + 1)))
    return [np.copy(individuo) for _ in range(num_clones)]

def aplicar_mutacao(anticorpo, taxa_mutacao):
    if np.random.rand() < taxa_mutacao:
        i, j = np.random.choice(len(anticorpo), size=2, replace=False)
        anticorpo[i], anticorpo[j] = anticorpo[j], anticorpo[i]

def algoritmo_clonal(matriz_distancias, tam_populacao, num_iteracoes, novas_solucoes, fator_clonagem):
    selecao_top = 10
    numero_cidades = len(matriz_distancias)

    populacao = criar_populacao(tam_populacao, numero_cidades)
    fitness_populacao = [calcular_fitness(anticorpo, matriz_distancias) for anticorpo in populacao]
    melhor_distancia = float('inf')
    historico_melhores = []

    for _ in range(num_iteracoes):
        indices_melhores = np.argsort(fitness_populacao)[-selecao_top:]
        melhores_solucoes = [populacao[i] for i in indices_melhores]
        melhores_fitness = [fitness_populacao[i] for i in indices_melhores]

        clones = []
        for i in range(selecao_top):
            clones += realizar_clonagem(melhores_solucoes[i], melhores_fitness[i], fator_clonagem, tam_populacao)

        for clone in clones:
            fitness_clone = calcular_fitness(clone, matriz_distancias)
            aplicar_mutacao(clone, np.exp(-fitness_clone))

        fitness_clones = [calcular_fitness(clone, matriz_distancias) for clone in clones]

        indices_clones_melhores = np.argsort(fitness_clones)[-selecao_top:]
        populacao = [clones[i] for i in indices_clones_melhores]

        populacao = expandir_populacao(populacao, novas_solucoes, numero_cidades)
        fitness_populacao = [calcular_fitness(anticorpo, matriz_distancias) for anticorpo in populacao]

        distancia_atual = 1 / max(fitness_populacao)
        if distancia_atual < melhor_distancia:
            melhor_distancia = distancia_atual

        historico_melhores.append(melhor_distancia)

    return historico_melhores

def salvar_grafico(historico, iteracoes, nome_arquivo, n, d, beta):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iteracoes), historico, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor distância')
    plt.title(fr'$n={n}$, $d= {d}$, $\beta = {beta}$')
    #plt.title(nome_arquivo.replace("_", " "))
    plt.tight_layout()
    plt.savefig(f'{nome_arquivo}.png', dpi=300)
    plt.close()

# Carregamento dos dados
numero_cidades, matriz_distancias = carregar_dados_cidades()

# Parâmetros fixos
num_iteracoes = 100
valores_n = [10, 20, 30]      # população
valores_d = [2, 5, 10]        # novas soluções
valores_beta = [1, 2, 5]      # fator de clonagem

os.makedirs("plots", exist_ok=True)

# Loop para testar todas combinações
for n in valores_n:
    for d in valores_d:
        for beta in valores_beta:
            print(f"Testando n={n}, d={d}, beta={beta}")
            historico = algoritmo_clonal(
                matriz_distancias=matriz_distancias,
                tam_populacao=n,
                num_iteracoes=num_iteracoes,
                novas_solucoes=d,
                fator_clonagem=beta
            )
            nome_arquivo = f"plots/evolucao_n{n}_d{d}_beta{beta}"
            salvar_grafico(historico, num_iteracoes, nome_arquivo, n, d, beta)

import numpy as np
import random
import sys

def calcular_makespan(sequencia, tempos):
    N = len(sequencia)
    M = len(tempos[0])
    C = np.zeros((N, M))

    for i, tarefa in enumerate(sequencia):
        for m in range(M):
            tempo_proc = tempos[tarefa][m]
            if i == 0 and m == 0:
                C[i, m] = tempo_proc
            elif i == 0:
                C[i, m] = C[i, m-1] + tempo_proc
            elif m == 0:
                C[i, m] = C[i-1, m] + tempo_proc
            else:
                C[i, m] = max(C[i-1, m], C[i, m-1]) + tempo_proc

    return C[-1, -1]

class Particula:
    def __init__(self, n_tarefas):
        self.pos = list(np.random.permutation(n_tarefas))
        self.best_pos = self.pos[:]
        self.best_fit = float('inf')

    def avaliar(self, tempos):
        return calcular_makespan(self.pos, tempos)

    def atualizar_best(self, tempos):
        fit = self.avaliar(tempos)
        if fit < self.best_fit:
            self.best_fit = fit
            self.best_pos = self.pos[:]

    def mover(self, gbest, w=0.5, c1=1.0, c2=1.0):
        nova_pos = self.pos[:]
        n = len(nova_pos)

        for i in range(n):
            if random.random() < c1:
                idx = self.best_pos.index(nova_pos[i])
                nova_pos[i], nova_pos[idx] = nova_pos[idx], nova_pos[i]
            if random.random() < c2:
                idx = gbest.index(nova_pos[i])
                nova_pos[i], nova_pos[idx] = nova_pos[idx], nova_pos[i]

        if random.random() < w:
            i, j = random.sample(range(n), 2)
            nova_pos[i], nova_pos[j] = nova_pos[j], nova_pos[i]

        self.pos = nova_pos

class PSO:
    def __init__(self, n_particulas=30, max_iter=100, w=0.3, c1=1.0, c2=1.0):
        self.N, self.M  = None, None
        self.tempos     = self.get_tempos()
        self.n_tarefas  = len(self.tempos)
        self.n_particulas = n_particulas
        self.max_iter   = max_iter
        self.w          = w
        self.c1         = c1
        self.c2         = c2

        self.particulas = [Particula(self.n_tarefas) for _ in range(n_particulas)]
        self.gbest = None
        self.gbest_fit = float('inf')

    def get_tempos(self): 
        with open(sys.argv[1], 'r') as f:
            N, M = map(int, f.readline().split())

            matriz = []
            for _ in range(N):
                linha = list(map(float, f.readline().split()))
                matriz.append(linha)

        self.N = N 
        self.M = M
        return matriz

    def rodar(self):
        for _ in range(self.max_iter):
            for p in self.particulas:
                p.atualizar_best(self.tempos)
                if p.best_fit < self.gbest_fit:
                    self.gbest_fit = p.best_fit
                    self.gbest = p.best_pos[:]

            for p in self.particulas:
                p.mover(self.gbest, self.w, self.c1, self.c2)

        return self.gbest, self.gbest_fit


# Instância de exemplo

pso = PSO(n_particulas=20, max_iter=100)
melhor_seq, melhor_makespan = pso.rodar()

print("Melhor sequência:", melhor_seq)
print("Melhor makespan:", melhor_makespan)

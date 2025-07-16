import numpy as np
import matplotlib.pyplot as plt

class Particula():
    def __init__(self, n, limites):
        self.x       = np.random.uniform(limites[0], limites[1], n)
        self.v       = np.random.uniform(-1, 1, n)
        self.pbest   = np.copy(self.x)
        self.mfitness= np.inf  

    def __str__(self):
        return f"X  = {self.x}\nV  = {self.v}\nPb = {self.pbest}\n"

class PSO(): 
    def __init__(self, n, m=40, c1=2.0, c2=.075, w=0.7, max_iter=100):
        self.limites = (-10, 10)
        self.m      = m
        self.c1     = c1
        self.c2     = c2
        self.w      = w 
        self.n      = n
        self.max_iter = max_iter

        self.pop    = self._gerar_pop()
        self.gbest  = np.copy(self.pop[0].pbest)
        self.best_fitness = self.fitness(self.pop[0].pbest)

    def _gerar_pop(self):
        return [Particula(self.n, self.limites) for _ in range(self.m)]

    def fitness(self, x):
        n = self.n
        x = np.asarray(x)
        part1 = -0.2 * np.sqrt(np.sum(x**2) / n)
        part2 = np.sum(np.cos(2 * np.pi * x)) / n
        return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e

    def rodar(self): 
        result_fitness = []

        for it in range(self.max_iter):
            print(f"Iteracao {it}/{self.max_iter}")
            for particula in self.pop:
                f = self.fitness(particula.x)

                #atualiza pbest
                if f < particula.mfitness:
                    particula.pbest = np.copy(particula.x)
                    particula.mfitness = f

                #atualiza gbest
                if f < self.best_fitness:
                    self.gbest = np.copy(particula.x)
                    self.best_fitness = f

            for particula in self.pop:
                r1 = np.random.rand(self.n)
                r2 = np.random.rand(self.n)

                #atualizacao da velocidade
                particula.v = (self.w*particula.v +self.c1*r1*(particula.pbest-particula.x) +self.c2*r2*(self.gbest-particula.x))

                #atualizacao da posicao
                particula.x = particula.x + particula.v

                #xij=xmin, se xij < xmin e xij=xmax, se xij > xmax
                particula.x = np.clip(particula.x, self.limites[0], self.limites[1])

            result_fitness.append(self.best_fitness)
        
            #print(f"It: {it+1}/{self.max_iter}\t\tMelhor fitness = {self.best_fitness:.10f}")
        return result_fitness

"""ks, melhor_fitness = [], []

for k in [10, 20, 50, 100, 200, 500, 1000, 200]: 
    meupso = PSO(n=10, m=200, max_iter=k)
    results = meupso.rodar()

    ks.append(k)
    melhor_fitness.append(meupso.best_fitness)

plt.figure(figsize=(10, 8), dpi = 400)
plt.scatter(x = ks , y=melhor_fitness)
plt.xlabel('Iterações')
plt.ylabel('Melhor Fitness')
plt.tight_layout()
plt.savefig('results.png', dpi = 400)"""


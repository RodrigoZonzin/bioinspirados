#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


class Particula():
    def __init__(self, n, limites):
        self.x       = np.random.uniform(limites[0], limites[1], n)
        self.v       = np.random.uniform(-1, 1, n)
        self.pbest   = np.copy(self.x)
        self.mfitness= np.inf  

    def __str__(self):
        return f"X  = {self.x}\nV  = {self.v}\nPb = {self.pbest}\n"


# In[3]:


class PSO(): 
    def __init__(self, n, m=40, c1=2.0, c2=2.0, w=0.7, max_iter=100):
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


# In[74]:


plt.figure(figsize=(10, 8), dpi = 400)


for j in range(10):
    ks, melhor_fitness = [], []
    for k in [5, 10, 20, 50, 70, 100, 150, 200, 400]: 
            meupso = PSO(n=10, m=k, max_iter=150)
            results = meupso.rodar()

            ks.append(k)
            melhor_fitness.append(meupso.best_fitness)

    plt.scatter(x = ks , y=melhor_fitness, label = f'Ensaio {j+1}')
#plt.plot(ks, melhor_fitness, ls= '-.', label = f'Ensaio {j+1}')
plt.xlabel('Número de Partículas')
plt.ylabel('Melhor Fitness')
plt.tight_layout()
plt.legend()

plt.savefig('variando_m.png', dpi = 400)
#plt.show()


# In[78]:


dados = []
for k in [5, 10, 20, 30, 70, 100, 200]: 
    for it in [100, 125, 150, 175, 200, 300, 400, 500]:
        meupso = PSO(n=10, m=k, max_iter=it)
        results = meupso.rodar()
        dados.append({'k': k, 'it': it, 'fitness': meupso.best_fitness})

df = pd.DataFrame(dados)


# In[79]:


tabela = df.pivot(index='it', columns='k', values='fitness')


# In[80]:


import seaborn as sns

plt.figure(figsize=(10, 7), dpi=400)
sns.heatmap(tabela, annot=True, cmap='cividis', fmt=".3f")
plt.xlabel("Número de partículas (m)")
plt.ylabel("Número de iterações")
plt.tight_layout()
plt.savefig("heatmap_fitness_novo.png", dpi=400)
plt.show()


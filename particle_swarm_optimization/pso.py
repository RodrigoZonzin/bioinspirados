import numpy as np

class Particula():
    def __init__(self, n):
        self.x       = np.ones(n)
        self.v       = np.ones(n)
        self.pbest   = np.ones(n)

        self.mfitness= 1

    def __str__(self):
        return f"X ={self.x}\nV ={self.v}\nPb={self.pbest}\n\n"
    

class PSO(): 
    def __init__(self, n, m = 40, c1 = 0.01, c2 = 0.01, w = 0.02):
        self.limites = (-3, 3)
        self.m      = m
        self.c1     = c1
        self.c2     = c2
        self.w      = w 

        self.n      = n             #tamanho da particula
        self.pop    = self._gerar_pop()
        self.gbest  = np.ones(n)

    
    def _gerar_pop(self):
        return [Particula(self.n) for _ in range(self.m)]
        
    def fitness(self, particula):
        x = particula.x
        n = self.n

        primeira_parte = -0.2 * np.sqrt(np.sum(x**2)/n)
        segunda_parte  = np.sum(np.cos(2 * np.pi * x))/n
        
        return -20 * np.exp(primeira_parte) - np.exp(segunda_parte) + 20 + np.exp(1)


    def rodar(self): 
        for i in range(self.m):
            if self.fitness(self.pop[i].x) < self.fitness(self.pop[i].p):
                self.pop[i].p = self.pop[i].x
            
                if self.fitness(self.pop[i].x) < self.fitness(self.pop[i].g)
            
        


meupso = PSO(m= 10)
[print(f) for f in meupso.pop]

import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class Gene:
    def __init__(self, alelos):
        self.alelos = alelos
        self.alelos_dim = len(self.alelos)

    def __str__(self):
        return str(self.alelos)

    def shape(self):
        return self.alelos_dim

def fitness(gene):
    x = gene.alelos
    n = len(x)
    sum1 = sum([xi ** 2 for xi in x])
    sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
    parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
    parte2 = -math.exp(sum2 / n)
    return parte1 + parte2 + 20 + math.e

# Create the 3D plot
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": "3d"})

# Create data
X = np.arange(-5, 5.25, 0.25)
Y = np.arange(-5, 5.25, 0.25)
X, Y = np.meshgrid(X, Y)

# Aqui criamos Z aplicando a função fitness em cada ponto (x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        gene = Gene([X[i, j], Y[i, j]])
        Z[i, j] = fitness(gene)

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis
ax.set_zlim(np.min(Z), np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Fitness')

plt.title('Função Fitness')
plt.savefig('funcAckley.png')

print(np.min(Z))
#plt.show()

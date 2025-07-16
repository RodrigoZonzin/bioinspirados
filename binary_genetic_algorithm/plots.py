import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


#Funcao de Ackley
def fitness(self, gene):
    x = self.bin_to_real(gene)
    n = len(x)

    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([math.cos(2 * math.pi * xi) for xi in x])
    parte1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
    parte2 = -math.exp(sum2 / n)
    return parte1 + parte2 + 20 + math.e


x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)

X, Y = np.meshgrid(x, y)

Z = fitness(X, Y)

# Criando o gráfico 3D para a elevação A(x, y)
fig = plt.figure(figsize=(12, 6))

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='plasma')
ax2.grid(False)  # Desativa o grid

plt.tight_layout()

plt.savefig('superficie_queimada.png')
plt.show()

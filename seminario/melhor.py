import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from FPA_ox import *

# Parâmetros de varredura
p_values = np.arange(0.1, 1.1, 0.2)
taxa_corte_values = np.arange(0.1, 1.1, 0.2)

# Matriz para armazenar os melhores makespans
resultados = np.zeros((len(p_values), len(taxa_corte_values)))

# Loop sobre combinações de parâmetros
for i, p in enumerate(p_values):
    for j, taxa_corte in enumerate(taxa_corte_values):
        fpa = FPA(p=p, taxa_corte=taxa_corte, _lambda=1.5, num_polens=60, maxIt=1000)
        fpa.rodar()
        resultados[i, j] = fpa.melhor_fitness  # ou fpa.melhor_fitness se esse for o nome

# Plot do heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(resultados, annot=True, fmt=".0f", xticklabels=np.round(taxa_corte_values, 1), yticklabels=np.round(p_values, 1), cmap="YlGnBu")
plt.xlabel("Taxa de Corte")
plt.ylabel("p")
#plt.title("Melhor Makespan obtido por combinação de parâmetros")
plt.tight_layout()
#plt.show()
plt.savefig('heatmap2.png', dpi = 400)

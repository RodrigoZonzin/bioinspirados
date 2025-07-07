import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from FPA_ox2 import *

# Listas para armazenar os valores
num_polens_list = list(range(1, 200))
melhores_fitness = []

# Avaliando o impacto de num_polens
for num_polens in num_polens_list: 
    fpa = FPA(num_polens=num_polens, maxIt=600)
    fpa.rodar()
    melhores_fitness.append(fpa.melhor_makespan)  # ou fpa.melhor_fitness

# Gráfico de haste (stem plot)
plt.figure(figsize=(12, 6))
(markerline, stemlines, baseline) = plt.stem(num_polens_list, melhores_fitness, use_line_collection=True)
plt.setp(markerline, marker='o', markersize=4)
plt.setp(stemlines, linewidth=0.8)

plt.xlabel("Número de Polens")
plt.ylabel("Melhor Fitness")
#plt.title("Influência do Número de Polens no Melhor Fitness")
#plt.grid(True)
plt.tight_layout()
plt.savefig('numPolens.png', dpi=400)

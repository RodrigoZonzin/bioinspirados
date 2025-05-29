import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lê o CSV (substitua pelo caminho correto)
df = pd.read_csv("resultados_ag_2.csv")

# Gráfico 1: Histograma da média de desempenho
plt.figure(figsize=(10, 6))
sns.histplot(df['Média'], kde=True, bins=10, color='skyblue')
plt.title("Histograma das Médias de Desempenho")
plt.xlabel("Média")
plt.ylabel("Frequência")
plt.grid(True)
plt.savefig('results1.png')
#plt.show()

# Gráfico 2: Comparação entre tipos de Seleção (roleta vs torneio)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Seleção', y='Melhor', data=df, palette="Set2")
plt.title("Comparação da Melhor Solução - Roleta vs Torneio")
plt.ylabel("Melhor Solução Encontrada")
plt.grid(False)
plt.savefig('results2.png')
#plt.show()

# Gráfico 3: Comparação entre operadores (OX vs CX)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Operador', y='Melhor', data=df, palette="Set2")
plt.title("Comparação da Melhor Solução - OX vs CX")
plt.ylabel("Melhor Solução Encontrada")
plt.grid(False)
plt.savefig('results3.png')
#plt.show()

# Gráfico 4: Interação entre Operador e Seleção
plt.figure(figsize=(10, 6))
sns.barplot(x='Operador', y='Melhor', hue='Seleção', data=df, ci='sd', palette="Set2")
plt.title("Melhor Solução por Operador e Tipo de Seleção")
plt.ylabel("Melhor Solução")
plt.grid(False)
plt.savefig('results4.png')
#plt.show()

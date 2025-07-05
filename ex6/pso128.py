from pso2 import *

meupso = PSO(n=128, m = 100, max_iter = 100000)
results = meupso.rodar()

plt.figure(figsize=(10, 8), dpi = 400)
plt.scatter(x = range(100000), y=results)
plt.xlabel('Iterações')
plt.ylabel('Fitness')
plt.savefig('rodando128.png', dpi = 400)
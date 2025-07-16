from FPA_ox import * 


plt.figure(figsize=(10, 8))

for es in range(10): 
    fpa = FPA(maxIt=1000)
    fpa.rodar()
    plt.scatter(range(fpa.maxIt), fpa.historico_fitness, label = f'Ensaio {es+1}')

    plt.xlabel('Iterações')
    plt.ylabel('Fitness')

plt.savefig('variasExecucoes05.png', dpi = 400)
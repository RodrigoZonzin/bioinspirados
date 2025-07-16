from FPA_ox import * 
from itertools import product

resultados = []

combinacoes = list(product(
    [0.1, 0.2, 0.5, 0.8, 0.9],     # p
    [0.2, 0.4, 0.6],               # taxa_corte
    [1.0, 1.5, 2.0],               # _lambda
    [50, 100, 200]                 # num_polens
))

for p, taxa_corte, _lambda, num_polens in combinacoes:
    print(f'Teste com p={p}, taxa_corte={taxa_corte}, _lambda={_lambda}, num_polens={num_polens}')
    
    fpa = FPA(
        num_polens=num_polens,
        taxa_corte=taxa_corte,
        p=p,
        _lambda=_lambda,
        maxIt=200  # Ajuste se quiser reduzir tempo de execução
    )
    
    fpa.rodar()

    resultados.append({
        'p': p,
        'taxa_corte': taxa_corte,
        'lambda': _lambda,
        'num_polens': num_polens,
        'melhor_makespan': fpa.melhor_fitness,
        'tempo_execucao': fpa.tempo_exec
    })

    # opcional: salva gráfico do histórico
    plt.figure(figsize=(10, 6))
    plt.plot(fpa.historico_fitness, label='Fitness')
    plt.title(f'p={p}, corte={taxa_corte}, λ={_lambda}, polens={num_polens}')
    plt.xlabel('Iteração')
    plt.ylabel('Makespan')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'grafico_p-{p}_corte-{taxa_corte}_lambda-{_lambda}_polens-{num_polens}.png', dpi=300)
    plt.close()

# Criar DataFrame
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('resultados_fpa_teste_combinatorio.csv', index=False)
#print(df_resultados.sort_values(by='melhor_makespan').head(10))  # Mostra os 10 melhores

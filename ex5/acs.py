import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx

class Formiga(): 
	def __init__(self, m, n):
		self.m = m
		self.n = n
		self.permutacao = self._gerar_caminho_aleatorio()
		self.cidadeAtual = -1

	def atualizaCidadeAtual(self, novaCidade): 
		self.cidadeAtual = novaCidade

	def _gerar_caminho_aleatorio(self): 
		return np.random.randint(low = 0, high=self.n, size=self.m)
	
	def __str__(self):
		return str(self.permutacao)
	
class ACS(): 
	def __init__(self, alpha, beta, epsilon = 2, rho=0.1 , Q=100, n_formigas=30, n_iteracoes=100):
		self.epsilon 	= epsilon
		self.alpha 		= alpha
		self.beta 		= beta 
		self.rho 		= rho 						#taxa de evaporacao
		self.Q 			= Q 						#constante para att de feromonio
		#self.m 			= n_formigas
		self.iteracoes 	= n_iteracoes
		self.matriz_d 	= self._gerar_matrix_distancia()
		self.m 			= self.matriz_d.shape[0]		
		self.n 			= self.matriz_d.shape[0]
		self.matriz_f 	= np.ones((self.n, self.n))  
		self.melhor_caminho 	= None
		self.melhor_distancia 	= np.inf

	def _gerar_matrix_distancia(self, arquivo='sgb128_dist.txt'): 
		return np.loadtxt(fname=arquivo)

	def _calcular_distancia(self, caminho): 
		distancia = 0
		for i in range(len(caminho) - 1): 
			distancia += self.matriz_d[caminho[i], caminho[i+1]]
		distancia += self.matriz_d[caminho[-1], caminho[0]]  			#retorno do ultimo ah origem 
		return distancia

	def _escolher_proxima_cidade(self, atual, visitadas): 
		probs = []
		for j in range(self.n):				#para cada cidade
			if j in visitadas:				
				probs.append(0)
			else:
				tau = self.matriz_f[atual, j] ** self.alpha			#calcula p_{ij}^k
				eta = (1 / self.matriz_d[atual, j]) ** self.beta
				probs.append(tau * eta)
		probs = np.array(probs)
		if probs.sum() == 0:
			probs = np.ones(self.n)
		probs /= probs.sum()										#normaliza
		
		return np.random.choice(range(self.n), p=probs)

	def _constroi_caminho(self): 
		caminhos = []
		distancias = []

		for _ in range(self.m):				#para cada formiga k
			caminho = []					#caminho da formiga k inicia vazio e eh posto em uma cidade aleatoria
			cidade_atual = np.random.randint(self.n) 
			caminho.append(cidade_atual)

			while len(caminho) < self.n:
				prox = self._escolher_proxima_cidade(cidade_atual, caminho)  #escolhe a proxima com base no caluclo p_{ij}^k
				caminho.append(prox)				
				cidade_atual = prox			

			dist = self._calcular_distancia(caminho)
			caminhos.append(caminho)
			distancias.append(dist)

			if dist < self.melhor_distancia:
				self.melhor_distancia = dist
				self.melhor_caminho = caminho
		
		return caminhos, distancias

	def _atualizar_feromonio(self, caminhos, distancias): 
		#tau_{ij} = (1-p)tau_{ij}+somatorio_{k=1, m} deltaTau_{ij}^k
		self.matriz_f *= (1 - self.rho)  					#evaporacao
		
		for caminho, dist in zip(caminhos, distancias): 
			for i in range(len(caminho)):
				de = caminho[i]
				para = caminho[(i + 1) % self.n]  			#volta inicio
				self.matriz_f[de, para] += self.Q/dist		#Q/L_k
				self.matriz_f[para, de] += self.Q/dist  	#simetria		

	def _atualizar_feromonio_elitismo(self, caminhos, distancias): 
		#tau_{ij} = (1-p)tau_{ij}+somatorio_{k=1, m} deltaTau_{ij}^k
		self.matriz_f *= (1 - self.rho)		#evaporacao

		for caminho, dist in zip(caminhos, distancias):
			for i in range(len(caminho)):
				de = caminho[i]
				para = caminho[(i + 1) % self.n]			#volta inicio
				self.matriz_f[de, para] += self.Q / dist	#Q/L_k			
				self.matriz_f[para, de] += self.Q / dist	#simetria

		# Reforço elitista do melhor caminho global
		for i in range(len(self.melhor_caminho)):
			de = self.melhor_caminho[i]
			para = self.melhor_caminho[(i + 1) % self.n]
			delta_elite = self.epsilon * (self.Q / self.melhor_distancia)
			self.matriz_f[de, para] += delta_elite
			self.matriz_f[para, de] += delta_elite

	def _cria_arestas(self, melhores_caminhos): 
		
		lista_arestas = []
		for caminho in melhores_caminhos: 
			arestas = []
			for i in range(len(caminho)-1): 
				#print((melhores_caminhos[i], melhores_caminhos[i+1]))
				arestas.append((caminho[i], caminho[i+1]))
			lista_arestas.append(arestas)
		return lista_arestas

	def printar_params(self): 
		print(f"{self.alpha} & {self.beta} & {self.epsilon} \\\\")

	def rodar(self): 
		print("inicio algoritmo\n")
		melhor_dist, melhores_caminhos = [], []
		
		for iteracao in range(self.iteracoes): 
			caminhos, distancias = self._constroi_caminho()
			self._atualizar_feromonio_elitismo(caminhos, distancias)
			melhor_dist.append(self.melhor_distancia)
			melhores_caminhos.append([int(x) for x in self.melhor_caminho])
			#print(f'{iteracao}:\t{self.melhor_distancia}\t{[int(x) for x in self.melhor_caminho]}')

			if iteracao == 99 and self.melhor_distancia == 291: 
				#print('Sim')
				self.printar_params()
				#print("*"*50)
				pass
			elif iteracao == 99 and not(self.melhor_distancia == 291): 
				#print('Não atingiu: ')
				#self.printar_params()
				#print("*"*50)
				pass
		print(self.iteracoes, melhor_dist)
		return melhor_dist, melhores_caminhos
	


"""nit = 150
plt.figure(figsize=(10, 8), dpi = 300)
for alpha in [1, 5, 10]: 
	for beta in [5]:
		for epsilon in [5]:
			acs = ACS(alpha=alpha, beta=beta, rho=0.5, Q=100, epsilon= epsilon, n_iteracoes=nit)
			dist, caminhos 	= acs.rodar()

			#plotando o fitness vs geracao
						
			plt.scatter(y=dist, x = range(0, nit), label = f'{alpha}, {beta}, {epsilon}')
#plt.legend()
plt.savefig(f'results_variandoAlpha.png')
plt.close()
"""

nit = 200
plt.figure(figsize=(8, 5), dpi = 300)	
alpha, beta, epsilon = 5, 10, 0.5

for ensaio in range(4): 
	print(f"Entrei no Ensaio {ensaio}")
	acs = ACS(alpha=alpha, beta=beta, rho=0.5, Q=100, epsilon= epsilon, n_iteracoes=nit)
	dist, caminhos 	= acs.rodar()
			
	
	plt.scatter(y=dist, x = range(0, nit), label = f'Ensaio: {ensaio}')
	plt.legend()

plt.savefig(f'sgb_results_ensaios_rodando1_{str(ensaio)}.png')
plt.close()


"""
alpha, beta, epsilon = 1, 1, 0.5
nit = 100

acs = ACS(alpha=alpha, beta=beta, rho=0.5, Q=100, epsilon= epsilon, n_iteracoes=nit)
dist, caminhos 	= acs.rodar()
arestas = acs._cria_arestas(caminhos)
print(arestas[-1])

g = nx.DiGraph()
for u in range(0, acs.n):
	for v in range(0, acs.n):
		if u != v: 
			g.add_edge(u, v)


pos =pos=nx.spring_layout(g, seed =42)

nx.draw(g, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
nx.draw_networkx_edges(
	g,
	pos,
	edgelist=arestas[-1],
	edge_color='red',
	width=2.5,
	arrows=True
)

nos_no_caminho = set([u for u, v in arestas[-1]] + [arestas[-1][-1][1]])
nx.draw_networkx_nodes(
	g,
	pos,
	nodelist=nos_no_caminho,
	node_color='orange',
	node_size=500
)

def extrair_caminho(arestas):
	if not arestas:
		return []
	
	caminho = [arestas[0][0]]  # começa com o primeiro nó de partida
	for u, v in arestas:
		caminho.append(v)
	return str(caminho)


plt.annotate(extrair_caminho(arestas[-1]),
	xy=(0.05, 0.95),
	xycoords='axes fraction',
	fontsize=10,
	verticalalignment='bottom',
	bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
)

plt.savefig('grafo.png', dpi = 300)
plt.show()
plt.close()
"""

def analisar_alpha():
	plt.figure(figsize=(8,5), dpi =300)
	alphas = [0.5, 5, 10]
	for alpha in alphas:
		acs = ACS(alpha=alpha, beta=5, rho=0.5, n_iteracoes=100)
		melhor_dist, _ = acs.rodar()
		plt.scatter(y=melhor_dist, x= range(100), label=rf'$\alpha$={alpha}')
	plt.axhline(291, color='black', linestyle='--', label='Ótimo Global (291)')
	plt.title(rf'Impacto de $\alpha$')
	plt.xlabel('Iterações')
	plt.ylabel('Melhor distância')
	plt.legend()
	#plt.grid()
	#plt.show()
	plt.savefig('resultsAnaliseAlpha.png')

def analisar_beta():
	plt.figure(figsize=(8,5), dpi =300)
	betas = [0.5, 5, 10]
	for beta in betas:
		acs = ACS(alpha=1, beta=beta, rho=0.5, n_iteracoes=100)
		melhor_dist, _ = acs.rodar()
		plt.scatter(y=melhor_dist, x= range(100), label=rf'$\beta=${beta}')
	plt.axhline(291, color='black', linestyle='--', label='Ótimo Global = 291')
	plt.title(rf'Impacto de $\beta$')
	plt.xlabel('Iterações')
	plt.ylabel('Melhor distância')
	plt.legend()
	#plt.grid()
	#plt.show()
	plt.savefig('resultsAnaliseBeta.png')

def analisar_rho():
	plt.figure(figsize=(8,5), dpi =150)
	rhos = [0.25, 0.5, 0.75]
	for rho in rhos:
		acs = ACS(alpha=1, beta=5, rho=rho, n_iteracoes=100)
		melhor_dist, _ = acs.rodar()
		plt.scatter(y=melhor_dist, x= range(100), label=rf'$\rho=${rho}')
	plt.axhline(291, color='black', linestyle='--', label='Ótimo Global = 291')
	plt.title(rf'Impacto de $\rho$')
	plt.xlabel('Iterações')
	plt.ylabel('Melhor distância')
	plt.legend()
	#plt.grid()
	#plt.show()
	plt.savefig('resultsAnaliseRho.png')

def analisar_iteracoes():
	iteracoes = [50, 100, 150]
	combinacoes = [
		{'alpha': 1.2, 'beta': 10, 'rho': 0.5},
		{'alpha': 1, 'beta': 1, 'rho': 0.5},
	]

	for i, params in enumerate(combinacoes):
		plt.figure(figsize=(8,5), dpi =300)
		for it in iteracoes:
			acs = ACS(alpha=params['alpha'], beta=params['beta'], rho=params['rho'], n_iteracoes=it, epsilon=5)
			melhor_dist, _ = acs.rodar()
			plt.plot(range(len(melhor_dist)), melhor_dist, label=f'{it} iterações')
			if melhor_dist[-1] == 291:
				print(f"Ótimo alcançado com {params} e {it} iterações.")
		#plt.axhline(291, color='black', linestyle='--', label='Ótimo Global (291)')
		plt.title(f'Impacto do número de iterações')
		plt.xlabel('Iterações')
		plt.ylabel('Melhor distância')
		plt.legend()
		#plt.grid()
		#plt.show()
		plt.savefig('resultsAnaliseiteracoes.png')


#analisar_alpha()
#analisar_beta()
#analisar_rho()
#analisar_iteracoes()

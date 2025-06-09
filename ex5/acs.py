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

	def _gerar_matrix_distancia(self, arquivo='lau15_dist.txt'): 
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
		print(f"alpha: {self.alpha}, beta: {self.beta}, epsilon: {self.epsilon}, rho:{self.rho}, Q:{self.Q}")

	def rodar(self): 
		melhor_dist, melhores_caminhos = [], []
		
		for iteracao in range(self.iteracoes): 
			caminhos, distancias = self._constroi_caminho()
			self._atualizar_feromonio_elitismo(caminhos, distancias)
			melhor_dist.append(self.melhor_distancia)
			melhores_caminhos.append([int(x) for x in self.melhor_caminho])
			#print(f'{iteracao}:\t{self.melhor_distancia}\t{[int(x) for x in self.melhor_caminho]}')

			if iteracao == 99 and self.melhor_distancia == 291: 
				print('Sim')
				self.printar_params()
				print("*"*50)
			elif iteracao == 99 and not(self.melhor_distancia == 291): 
				print('Não atingiu: ')
				self.printar_params()
				print("*"*50)

		return melhor_dist, melhores_caminhos
	

plt.figure(figsize=(10, 8), dpi = 300)
nit = 100
for alpha in [0.5, 1, 1.2]: 
	for beta in [0.5, 1, 5, 10]:
		for epsilon in [0.01, 0.1, 0.5, 1, 3]:
			acs = ACS(alpha=alpha, beta=beta, rho=0.5, Q=100, epsilon= epsilon, n_iteracoes=nit)
			dist, caminhos 	= acs.rodar()

			#plotando o fitness vs geracao			
			plt.scatter(y=dist, x = range(0, nit), label = f'{alpha}, {beta}, {epsilon}')
			plt.savefig('results.png')
			plt.legend()





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
	def __init__(self, alpha, beta, m =100):
		self.m = m                              				#m = numero de formigas
		self.alpha = alpha
		self.beta = beta 
		self.matriz_d = self._gerar_matrix_distancia()         	#uma matriz nxn, com as distancias das cidades
		self.n = self.matriz_d.shape[0]
		self.matriz_f = self._gerar_matrix_feromonio()         	#uma matriz nxn, com os feromonios de cada aresta
		self.formigas = self._gerar_formigas()

	def _gerar_matrix_distancia(self, arquivo = 'lau15_dist.txt'): 
		return np.loadtxt(fname=arquivo)

	def _gerar_matrix_feromonio(self):
		return np.random.rand(self.n, self.n)*10		#gera valores aleatorios com shape nxn

	def print_matrix(self, param: str): 
		if param == 'f': 
			print(self.matriz_f)
		else: 
			print(self.matriz_d)

	def _gerar_formigas(self): 
		return [Formiga(self.m, self.n) for _ in range(self.m)]
	
	def printar_formigas(self): 
		[print(f) for f in self.formigas]

	def probabilidade_nova_cidade(self, formiga: Formiga, i, j): 
		numerador = (self.matriz_f[i, j]**self.alpha) * ((1/self.matriz_d)**self.beta)		#tau_{ij}^a*etha_{ij}^beta
		denominador  = [(self.matriz_f[i, k]**self.alpha) * ((1/self.matriz_d[i, k])**self.beta) for k in range(0, self.n) if k != j]
		print(numerador/denominador)

acs = ACS(1, 2)
acs.print_matrix('')
acs.printar_formigas()
#acs.print_matrix('f')
#f1 = Formiga(2, 3)
#print(f1)



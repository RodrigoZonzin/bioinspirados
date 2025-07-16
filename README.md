# Bio-inspired computing for optimization

The following optimization algorithms were implemented from scratch (with NumPy). 

1. Genetic Algorithm (binary, continuous and combinatorial)
2. Particle Swarm Optimization Algorithm 
3. Ant Colony System
4. Clonal Selection 
5. Flower Pollination Algorithm (my favourite)

## Genetic Algorithm 
### Ackley
The n-dimensional Ackley function is defined by the following equation: 

$$ f(\mathbf{x}) = -20 e^{-0.2 \sqrt{\frac{1}{n} \sum \mathbf{x}_i ^2}} -e^{\frac{1}{n}\sum cos(2 \pi \mathbf{x}_i)} +20 +e $$

<img src="continuous_genetic_algorithm/funcAckley.png" width="400" style="display: block; margin: auto"/>

#### Results
Using parameter optimization, the following results were obtained: 

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="continuous_genetic_algorithm/pr2_bioinspirados_rodrigojzonzin/results/fitness_conjunto.png" width="400"/>
  <img src="continuous_genetic_algorithm/pr2_bioinspirados_rodrigojzonzin/resultsMutacao/fitness_mutacao.png" width="400"/>
</div>

### Travelling Salesman Problem
Classic combinatorial optimization problem that aims to find the shortest possible route that visits a set of cities exactly once and returns to the starting point. Given a permutation $\pi$ of $n$ cities, the total distance is defined by:

$$f(\pi)=\sum_{i=1}^{n-1} \rho(\pi(i), \pi(i+1))+\rho(\pi(n), \pi(1))$$

where  $\rho(a, b)$ represents the distance between cities $a$ and $b$. For testing (lau_15.txt instance), the global minimun is known as $f(\pi_0) = 291$. 

#### Results

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="combinatorial_genetic_algorithm/results_ter_ox.png" width="400"/>
  <img src="combinatorial_genetic_algorithm/results2.png" width="400"/>
</div>
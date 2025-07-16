# Bio-inspired computing for optimization

The following optimization algorithms were implemented from scratch (with NumPy). 

1. Genetic Algorithm (binary, continuous and combinatorial)
2. Particle Swarm Optimization Algorithm 
3. Ant Colony System
4. Clonal Selection 
5. Flower Pollination Algorithm (my favourite)

## GA for Ackley 
The n-dimensional Ackley function is defined by the following equation: 

$$ f(\mathbf{x}) = -20 e^{-0.2 \sqrt{\frac{1}{n} \sum \mathbf{x}_i ^2}} -e^{\frac{1}{n}\sum cos(2 \pi \mathbf{x}_i)} +20 +e $$

<img src="continuous_genetic_algorithm/funcAckley.png" width="400" style="display: block; margin: auto"/>

Results: 

<img src="continuous_genetic_algorithm/pr2_bioinspirados_rodrigojzonzin/results/fitness_conjunto.png" width="400" style="display: block; margin: auto"/>
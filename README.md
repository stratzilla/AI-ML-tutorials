# Tutorials

This is a collection of machine learning and artificial intelligence tutorials I've written using Jupyter Notebook. They may not be up to snuff with some other tutorials, and in many cases, I've written them after only one or two implementations. I'm keeping these here for posterity for myself and for those who might also be interested. I had a hard time finding tutorials online for some of these subjects, so I thought I'd make my own for those who find themselves in a similar position to what I was.

I try my best to only use Python standard libraries, although I do quite like `numpy`, `pandas`, and `matplotlib`. Some `gnuplot` is involved as well but only for illustrative purposes. All of these implementations are from scratch insofar I don't rely on external libraries to do any of the heavy work. External libraries might be used for things like IO or as visual tools instead.

Feel free to distribute, copy, edit, make changes to, etc. these tutorials to your hearts content. All I ask is if you publish any of them elsewhere, please let me know and link back.

### <a href="./genetic-algorithms">Genetic Algorithms</a>

A step-by-step implementation of a genetic algorithm to optimize the Styblinski-Tang function, generalized to any dimensionality. Includes one-point crossover and uniform mutation operators, tournament and elite selection strategies.

### <a href="./particle-swarm-optimization">Particle Swarm Optimization</a>

A step-by-step implementation of a particle swarm algorithm to optimize the Schwefel function, generalized to any dimensionality.

### <a href="./k-means">K-Means Clustering</a>

A step-by-step implementation of a K-Means clustering algorithm using K-Means++ initialization strategy. Includes Dunn Index calculation to determine merit in cluster count optimization.

### <a href="./neural-network">Neural Network using Backpropagation Training</a>

A step-by-step implementation of a vanilla feedforward neural network trained via backpropagation to model a classifier for the Iris data set. Includes learning rate and momentum hyperparameters.

### <a href="./genetic-neural-network">Neural Network using Genetic Algorithm Training</a>

A step-by-step implementation of a feedforward neural network, but instead trained using a genetic algorithm to optimize network weights to model a classifier for the Wheat Seeds data set. It helps to have read the <i>Neural Network using Backpropagation Training</i> and <i>Genetic Algorithms</i> tutorials beforehand.

### <a href="./particle-neural-network">Neural Network using Particle Swarm Optimization Training</a>

A step-by-step implementation of a feedforward neural network, but instead using a particle swarm algorithm to optimize neural weights to model a classifier for the Wine data set. It helps to have read the <i>Neural Network using Backpropagation Training</i> and <i>Particle Swarm Optimization</i> tutorials beforehand.

## Stuff I'm Working On

Eventually I hope to add these tutorials at some point (in no particular order):

- color organization using Self Organizing Maps (SOMs)
- simulated annealing metaheuristic
- differential evolution metaheuristic
- training a neural network using differential evolution
- bat algorithm metaheuristic
- training a neural network using bat algorithm

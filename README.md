# Markov-Random-Field
First we simulated 5x5 grid binary Markov Random Field, simulating x using Gibbs sampling, then creating y: y_i|x_i ~ N(x_i,1).<br />
Then we computed the correct marginal p(x_i|y), and plotted the error of the approximate probabilities in comparison to the correct probabilities.<br />
The approximation was computed by Gibbs sampling algorithm and by Mean-Field approximation.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functions_task2_1 import phi, grad_phi, sigma, cost_func, grad_1_cost, grad_2_cost, animation, move_intruders, plot_cost, plot_trajectories, moving_targets_animation

### Parameters ###
# Number of agents
NN = 6
# Learning rate
alpha = 0.01
# Number of iterations
MAXITERS = 10000
# Initialize the central target
r_0 = np.array([2.5, 2.5]) 
# for moving intruders set this to True
MOVE_R_I = True

# Initialize positions of agents and intruders

d = 2 # Dimensionality of the space (2D in this case)
np.random.seed(42)
z = np.random.uniform(0, 5, (NN, d))
r_i = np.random.uniform(0, 5, (NN, d))


# Adjacency matrix for a path graph
G = nx.path_graph(NN)
Adj = nx.adjacency_matrix(G).toarray()

# Weights for consensus (normalized adjacency matrix)
AA = np.zeros((NN, NN))
for i in range(NN):
    neighbors = np.nonzero(Adj[i])[0]
    deg_i = len(neighbors)
    for j in neighbors:
        deg_j = len(np.nonzero(Adj[j])[0])
        AA[i, j] = 1 / (1 + max(deg_i, deg_j))
AA += np.eye(NN) - np.diag(np.sum(AA, axis=0))



# Initializing parameters for the cost function
cc = np.ones(NN) *1.0001
beta = np.ones(NN) *  0.0 * 0.1      # pursue r_0 with barycenter
delta = np.ones(NN) * 0.0 * 0.1     # go to own barycenter
gamma = np.ones(NN) * 2.0 * 0.1     # go to intruder

# Initialize the arrays
ZZ = np.zeros((MAXITERS, NN, d))
SS = np.zeros((MAXITERS, NN, d))
VV = np.zeros((MAXITERS, NN, d))
cost = np.zeros(MAXITERS)

norm_grad = np.zeros(MAXITERS)
grad_cost = np.zeros((MAXITERS, NN, d))
sigma_plot = np.zeros((MAXITERS, d))


# Initialize the positions of the agents
ZZ[0] = z



r_i_history = np.zeros((MAXITERS, NN, d))
r_i_initial = r_i

## Update loop for the agents through aggregative optimization
for kk in range(MAXITERS - 1):

    if MOVE_R_I:
        r_i = move_intruders(r_i, kk, r_i_initial)
        r_i_history[kk] = r_i

    if kk == 0:
        # calculating the initial values of S_i and V_i
        for ii in range(NN):
            SS[0, ii] = phi(cc[ii], ZZ[0, ii])
            VV[0, ii] = grad_2_cost(ZZ[0, ii], SS[0, ii], r_0, beta[ii], delta[ii])
   
    # calculating the sigma_z for cost and gradient calculation
    sigma_z = sigma(cc, ZZ[kk])
    sigma_plot[kk] = sigma_z

    # Update the agents
    for ii in range(NN):
        ZZ[kk + 1, ii] = ZZ[kk, ii] - alpha * (grad_1_cost(ZZ[kk, ii], r_i[ii], SS[kk, ii], gamma[ii], delta[ii]) + grad_phi(cc[ii], ZZ[kk, ii]) @ VV[kk, ii])
        SS[kk + 1, ii] = phi(cc[ii], ZZ[kk + 1, ii]) - phi(cc[ii], ZZ[kk, ii])
        for jj in range(NN):
            SS[kk + 1, ii] += AA[ii, jj] * SS[kk, jj]
        VV[kk + 1, ii] = grad_2_cost(ZZ[kk + 1, ii], SS[kk + 1, ii], r_0, beta[ii], delta[ii]) - grad_2_cost(ZZ[kk, ii], SS[kk, ii], r_0, beta[ii], delta[ii])
        for jj in range(NN):
            VV[kk + 1, ii] += AA[ii, jj] * VV[kk, jj]

        # Calculate the cost
        cost[kk] += cost_func(ZZ[kk, ii], r_i[ii], sigma_z, r_0, gamma[ii], beta[ii], delta[ii])

        grad_cost[kk, ii] = grad_1_cost(ZZ[kk, ii], r_i[ii], sigma_z, gamma[ii], delta[ii]) + (1/NN) * grad_phi(cc[ii], ZZ[kk, ii]) @ grad_2_cost(ZZ[kk, ii], sigma_z, r_0, beta[ii], delta[ii])
        
    norm_grad[kk] = np.linalg.norm(grad_cost[kk])
    if norm_grad[kk] < 1e-6:
        print(f'Converged at iteration {kk}')
        MAXITERS = kk
        break

# Plot the costs
plot_cost(cost, norm_grad, MAXITERS)

# Plot the trajectories
plot_trajectories(ZZ, r_i_initial, r_0, MAXITERS, NN, MOVE_R_I, r_i_history, cc)

# Call the animation function
if MOVE_R_I:
    moving_targets_animation(ZZ, NN, d, MAXITERS, AA, r_i_history, sigma_plot, MOVE_R_I, r_i_initial, r_0)
else:
    animation(ZZ, NN, d, MAXITERS, AA, r_i_initial, r_0, sigma_plot)
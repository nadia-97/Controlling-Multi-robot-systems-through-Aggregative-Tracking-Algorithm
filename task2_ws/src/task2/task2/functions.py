import numpy as np


def phi(cc, z):
    cc = np.eye(len(z)) * cc
    return cc @ z

def grad_phi(cc, z):
    cc = np.eye(len(z)) * cc
    return cc

# sigma gives the barycentre of the agents
def sigma(cc, z):
    transformed_z = np.array([phi(cc, z[i]) for i in range(len(z))])
    return np.mean(transformed_z, axis=0)

# Cost function for each agent
def cost_func(z_i, r_i, sigma_z, r_0, gamma_i, beta_i, delta_i):
    target_proximity = gamma_i * np.linalg.norm(z_i - r_i) ** 2
    barycenter_term = beta_i * np.linalg.norm(sigma_z - r_0) ** 2
    formation_tightness = delta_i * np.linalg.norm(z_i - sigma_z) ** 2
    return formation_tightness + target_proximity + barycenter_term

# Gradients of the cost function
def grad_1_cost(z_i, r_i, s_i, gamma_i, delta_i):
    return 2 * gamma_i * (z_i - r_i) + 2 * delta_i * (z_i - s_i)

def grad_2_cost(z_i, s_i, r_0, beta_i, delta_i):
    return 2 * beta_i * (s_i - r_0) - 2 * delta_i * (z_i - s_i)


# Function to project agents to the corridor
def project_to_corridor(z_i, corridor_y, corridor_x, safe_distance=0.1):
    if z_i[0] >= corridor_x[0] - 0.3 and z_i[0] <= corridor_x[1] + 0.3:
        if z_i[1] < corridor_y[0] + safe_distance:
            z_i[1] = corridor_y[0] + safe_distance
        elif z_i[1] > corridor_y[1] - safe_distance:
            z_i[1] = corridor_y[1] - safe_distance
    
    return z_i

## this is a barrier function to compute the cost of the distance between two agents 

def inter_agent_distance_cost(z_i, z_j, safe_distance=0.01, collision_avoidance=True):
    if collision_avoidance:
        distance_squared = np.dot(z_i - z_j, z_i - z_j)
        argument = distance_squared - safe_distance**2
        if argument <= 0:  # Avoid log of non-positive numbers
            return 1000      # or some large value to indicate error and to avoid division by zero
        result = -np.log(argument)
    else:
        result = 0
    return result

## this is the gradient of the barrier function 

def inter_agent_distance_grad(z_i, z_j, safe_distance=0.01, collision_avoidance=True):
    if collision_avoidance:
        distance_squared = np.dot(z_i - z_j, z_i - z_j)
        argument = distance_squared - safe_distance**2
        gradient = -2 * (z_i - z_j) / argument
        for i in range(len(z_i)):
            if gradient[i] == np.inf:
                return 1000 #to avoid inf values and errors in python
            if gradient[i] == -np.inf:
                return -1000 #to avoid inf values and errors in python
    else:
        gradient = np.zeros(len(z_i))
    return gradient

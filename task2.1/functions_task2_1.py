import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

blue_O4S = mcolors.to_rgb((0 / 255, 41 / 255, 69 / 255))
emph_O4S = mcolors.to_rgb((0 / 255, 93 / 255, 137 / 255))
red_O4S = mcolors.to_rgb((127 / 255, 0 / 255, 0 / 255))
gray_O4S = mcolors.to_rgb((112 / 255, 112 / 255, 112 / 255))

#gives weights to the agents positions
def phi(cc, z):
    cc = np.eye(len(z)) * cc
    return cc @ z
# Gradient of the function phi
def grad_phi(cc, z):
    cc = np.eye(len(z)) * cc
    return cc

# Computes Barycenter of the agents positions
def sigma(cc, z):
    transformed_z = np.array([phi(cc[i], z[i]) for i in range(len(z))])
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

# Define the motion model for the moving target r_i

def move_intruders(r_i, kk, center):
    for i in range(len(r_i)):
        r_i[i][0] = center[i][0] + 0.001 * np.cos(0.01 * kk)  # radius = 0.001, frequency = 0.01 for a cicrular motion model
        r_i[i][1] = center[i][1] + 0.001 * np.sin(0.01 * kk)
    return r_i

# Animation function of the agents and barycenters
def animation(ZZ, NN, d, MAXITERS, AA, r_i, r_0, sigma):
    overall_min = np.min(ZZ) - 1
    overall_max = np.max(ZZ) + 1

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(overall_min, overall_max)
    ax.set_ylim(overall_min, overall_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    # Initialize the plot objects for agents and barycenters
    intruders_plots = [ax.plot([], [], 's', color='red', label='Intruders (r_i)')[0] for _ in range(NN)]
    agents_plots = [ax.plot([], [], 'o', color='blue', markersize=5, label='Agents')[0] if i == 0 else ax.plot([], [], 'o', color='blue', markersize=5)[0] for i in range(NN)]
    barycenter_r0_plot, = ax.plot([], [], '^', color='black', markersize=5, label='Barycenter r_0')
    barycenter_sigma_plot, = ax.plot([], [], 'o', color='green', markersize=5, label='Final r_0')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    for tt in range(MAXITERS):
        if tt % 10 == 0:
            # Update the positions of the intruders
            for i in range(NN):
                intruders_plots[i].set_data(r_i[i, 0], r_i[i, 1])
                agents_plots[i].set_data(ZZ[tt, i, 0], ZZ[tt, i, 1])

            # Update the barycenters
            barycenter_r0_plot.set_data(r_0[0], r_0[1])
            barycenter_sigma_plot.set_data(sigma[tt][0], sigma[tt][1])

            title.set_text(f"Agent positions - Time step = {tt}")
            plt.draw()
            plt.pause(0.1)

    plt.show()  # Only call plt.show() once, after the loop has finished
    plt.close(fig)  # Close the figure to prevent empty figures


# Animation function of the agents and barycenters in case of moving intruders

def moving_targets_animation(ZZ, NN, d, MAXITERS, AA, r_i_history, sigma, MOVE_R_I, r_i, r_0):
    overall_min = np.min(ZZ) - 1
    overall_max = np.max(ZZ) + 1

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(overall_min, overall_max)
    ax.set_ylim(overall_min, overall_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    # Initialize the plot objects for agents and barycenters
    intruders_plots = [ax.plot([], [], 's', color='red', label='Intruders (r_i)')[0] for _ in range(NN)]
    agents_plots = [ax.plot([], [], 'o', color='blue', markersize=5, label='Agents')[0] if i == 0 else ax.plot([], [], 'o', color='blue', markersize=5)[0] for i in range(NN)]
    barycenter_r0_plot, = ax.plot([], [], '^', color='black', markersize=5, label='Barycenter r_0')
    barycenter_sigma_plot, = ax.plot([], [], 'o', color='green', markersize=5, label='Final r_0')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    for tt in range(MAXITERS):
        if tt % 10 == 0:
            # Update the intruders positions from the histories
            for i in range(NN):
                if MOVE_R_I:
                    r_i = r_i_history[tt, i]
                    intruders_plots[i].set_data(r_i[0], r_i[1])
                else: 
                    intruders_plots[i].set_data(r_i[i, 0], r_i[i, 1])
               
                agents_plots[i].set_data(ZZ[tt, i, 0], ZZ[tt, i, 1])
           
            barycenter_r0_plot.set_data(r_0[0], r_0[1])
                # Update the barycenter positions from the sigma history
                

            if tt < len(sigma):
                barycenter_sigma_plot.set_data(sigma[tt][0], sigma[tt][1])

            title.set_text(f"Agent positions - Time step = {tt}")
            plt.draw()
            plt.pause(0.1)

    plt.show() 
    plt.close(fig)  


def plot_cost(cost, norm_grad, MAXITERS):
    # Plotting the results
    plt.subplot(1, 2, 1)
    plt.plot(cost[:MAXITERS - 1])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost function over iterations')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(norm_grad[:MAXITERS - 1])
    plt.xlabel('Iterations')
    plt.ylabel('Norm of the gradient')
    plt.title('Norm of the gradient over iterations')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    plt.show()

def plot_trajectories(ZZ, r_i, r_0, MAXITERS, NN, move_r_i, r_i_history, cc):
    
    plt.figure()
    # Plot initial positions of agents
    plt.scatter(ZZ[0, :, 0], ZZ[0, :, 1], color='blue', label='Initial Positions', marker='o')

    # Plot targets (intruders' positions)
    plt.scatter(r_i[:, 0], r_i[:, 1], color='red', label='Intruders (r_i)', marker='s')

    # Plot initial and final positions of the central target
    
    plt.scatter(r_0[0], r_0[1], color='green', marker='x', label='Central Target')
    if move_r_i:
        for ii in range(NN):
            plt.plot(r_i_history[:MAXITERS-1, ii, 0], r_i_history[:MAXITERS-1, ii, 1], 'r--', label='Intruders Trajectory')
            # plt.scatter(r_i_history[0, ii, 0], r_i_history[0, ii, 1], color='red', marker='x', label='Initial r_i')
            # plt.scatter(r_i_history[-2, ii, 0], r_i_history[-2, ii, 1], color='red', marker='o', label='Final r_i')

    # Plot final positions and trajectories of agents
    for ii in range(NN):
        plt.plot(ZZ[:MAXITERS-1, ii, 0], ZZ[:MAXITERS-1, ii, 1])

    # Plot barycenter over iterations
    barycenters = np.array([sigma(cc, ZZ[k]) for k in range(MAXITERS-1)])
    plt.plot(barycenters[:, 0], barycenters[:, 1], 'k--', label='Barycenter Trajectory')
    plt.scatter(barycenters[0, 0], barycenters[0, 1], color='black', marker='^', label='Initial Barycenter')
    plt.scatter(barycenters[-1, 0], barycenters[-1, 1], color='black', marker='o', label='Final Barycenter')

    # Final positions of agents
    plt.scatter(ZZ[MAXITERS-1, :, 0], ZZ[MAXITERS-1, :, 1], color='blue', marker='^', label='Final Positions')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Agent trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

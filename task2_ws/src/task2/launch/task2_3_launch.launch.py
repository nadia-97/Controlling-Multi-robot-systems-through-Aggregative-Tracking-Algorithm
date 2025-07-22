from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
from networkx import neighbors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os
from ament_index_python.packages import get_package_share_directory

def generate_weights(NN, Adj):
    I_NN = np.eye(NN)
    AA = np.zeros(shape=(NN, NN))
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]
        deg_ii = len(N_ii)
        for jj in N_ii:
            N_jj = np.nonzero(Adj[jj])[0]
            deg_jj = len(N_jj)
            AA[ii, jj] = 1/(1 + max([deg_ii, deg_jj])) 

    AA += I_NN - np.diag(np.sum(AA, axis=0))

    return AA

# Define room dimensions
room1_x = [1.0, 3.0]
room2_x = [7.0, 8.0]
room1_y = [0.5, 3.5]
room2_y = [1.5, 2.5]
corridor_x = [3.0, 6.0]
corridor_y = [1.5, 2.5]

# corridor_length = corridor_x[1] - corridor_x[0]
# corridor_width = corridor_y[1] - corridor_y[0]

NN=5
Enable_Collision_Avoidance = True

if Enable_Collision_Avoidance:
    G = nx.complete_graph(NN)
    print("Since Collision Avoidance is enabled, the graph is a complete graph.")
else:
    G = nx.path_graph(NN)
    print("Since Collision Avoidance is disabled, the graph can be a path graph.")

# check if G is complete graph


Adj = nx.adjacency_matrix(G).toarray()
AA = generate_weights(NN, Adj)
d = 2

# Initialize positions of agents and targets
np.random.seed(42)

zzero = np.zeros((NN, d))
for i in range(NN):
    zzero[i,0] = np.random.uniform(room1_x[0],room1_x[1])   # Agents start from the left side of the corridor
    zzero[i,1] = np.random.uniform(room1_y[0],room1_y[1])     

r_i = np.zeros((NN, d))
for i in range(NN):
    r_i[i,0] = np.random.uniform(room2_x[0],room2_x[1]) # Random x position within the right room
    r_i[i,1] = np.random.uniform(room2_y[0],room2_y[1])

r_0 = np.array([room2_x[1] - room2_x[0], room2_y[1] - room2_y[0]]) / 2 + np.array([room2_x[0], room2_y[0]]) # Central target initial

# scale = 5
# zzero = scale*np.random.uniform(size=(NN, 2)) #starting coordinates
# target = scale*np.random.uniform(size=(NN, 2)) #coordinates of target

r_i = r_i
r_0 = r_0

cc = 1.0001
beta = 0.0 * 0.1          # takes barycenter to r_0
delta = 0.0 * 0.1          # goes to barycenter
gamma = 10.0 *0.1           # goes to r_i intruder
alpha = 0.03
MAXITERS = 600

zeta = 0.3
inter_agent_safe_distance = 0.1
corridor_safe_distance = 0.11

target = (zzero + np.random.uniform(0, 0.5, (NN, d)))
print(f'zzero: {zzero}')
print(f'target: {target}')

def generate_launch_description(): 
    node_list = []

    for i in range(NN): 
        A_ii = AA[i,:]                          #the i-th row of the matrix AA corresponding only to agent i and its neighbors
        N_ii = np.nonzero(A_ii)[0]              #number of neighbors of agent i
        zzero_i = zzero[i, :]                   #coordinates of starting point of i
        print(f'agent {i}, zzero: {zzero_i}')
        # r_i = target[i, :]                    #convert numpy array to list
        r_i_i = r_i[i, :]
        # print(f'agent {i}, target: {r_i}')
        
        node_list.append(
            Node(
                package='task2',
                namespace=f"agent_{i}", 
                executable='task2_3',           #the executable file which each agent will run
                parameters=[                    #parameters passed to each agent at launch
                    {"id":i,
                     "A_ii":A_ii.tolist(),
                     "N_ii":N_ii.tolist(), 
                     "zzero":zzero_i.tolist(), 
                     "r_i":r_i_i.tolist(),
                        "MAXITERS":MAXITERS,
                        "r_0":r_0.tolist(),
                        "cc":cc,
                        "beta":beta,
                        "delta":delta,
                        "gamma":gamma,
                        "alpha":alpha,
                    "corridor_x":corridor_x,
                    "corridor_y":corridor_y,
                    "inter_agent_safe_distance":inter_agent_safe_distance,
                    "Collision_Avoidance":Enable_Collision_Avoidance,
                    "corridor_safe_distance":corridor_safe_distance,
                    "zeta":zeta
                    }
                ],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )

    node_list.append(
    Node(
        package='task2',
        executable='task2_3_plotter',
        output='screen',
        parameters=[
            {"NN":NN ,
                "MAXITERS":MAXITERS, 
                "r_0":r_0.tolist(),
                "cc":cc,
                "beta":beta,
                "delta":delta,
                "gamma":gamma,
            "AA":AA.flatten().tolist(),
            "AA_shape":AA.shape,
            "zzero":zzero.flatten().tolist(),
            "zzero_shape":zzero.shape,
            "r_i":r_i.flatten().tolist(),
            "r_i_shape":r_i.shape,
            "d":d,
            "corridor_x":corridor_x,
            "corridor_y":corridor_y,
            "Collision_Avoidance":Enable_Collision_Avoidance
            }
        ],
        prefix=f'xterm -title "task2_visualizer" -hold -e',
    )
)

    return LaunchDescription(node_list)


    
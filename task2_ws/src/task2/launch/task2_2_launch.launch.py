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
        N_ii = np.nonzero(Adj[ii])[0] #neighbors of agent ii
        deg_ii = len(N_ii) #degree of agent ii
        for jj in N_ii:
            N_jj = np.nonzero(Adj[jj])[0]
            deg_jj = len(N_jj)
            AA[ii, jj] = 1/(1 + max([deg_ii, deg_jj]))  #MH weights

    AA += I_NN - np.diag(np.sum(AA, axis=0)) 

    return AA

NN=5
G=nx.path_graph(NN)
Adj = nx.adjacency_matrix(G).toarray()
AA = generate_weights(NN, Adj)
d = 2


zzero = np.random.uniform(0, 5, (NN, d))
r_i = np.random.uniform(0, 5, (NN, d))
print(f'zzero: {zzero}')
print(f'r_i: {r_i}')


MAXITERS = 300
r_0 = np.array([4.5, 4.5])
cc = 1.0001
beta = 10.0 * 0.1          # takes barycenter to r_0
delta = 10.0 * 0.1          # goes to barycenter
gamma = 0.0 *0.1           # goes to r_i intruder
alpha = 0.01
def generate_launch_description(): 
    node_list = []

    for i in range(NN): 
        A_ii = AA[i,:] #the i-th row of the matrix AA corresponding only to agent i and its neighbors
        N_ii = np.nonzero(A_ii)[0] #number of neighbors of agent i
        zzero_i = zzero[i, :] #coordinates of starting point of i
        r_i_i = r_i[i, :] # coordiantes of r_i of i
        print(f'agent {i}, r_i: {r_i_i}')
        
        node_list.append(
            Node(
                package='task2',
                namespace=f"agent_{i}",
                executable='task2_2', #the executable file which each agent will run
                parameters=[ #parameters passed to each agent at launch 
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


                     }
                ],
                output='screen', 
                prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )
    # AA_flatten = AA.flatten()
    node_list.append(
        Node(
            package='task2',
            executable='task2_2_plotter',
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
                }
            ],
            prefix=f'xterm -title "task2_visualizer" -hold -e',
        )
    )
  
    return LaunchDescription(node_list)


    
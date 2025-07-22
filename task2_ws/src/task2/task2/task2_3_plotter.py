import rclpy
from rclpy.node import Node
import scipy.sparse as sp
from time import sleep 
import numpy as np
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
import rclpy.time
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray as MsgFloat

from task2.functions import phi, grad_phi, sigma, cost_func, grad_1_cost, grad_2_cost, inter_agent_distance_cost, inter_agent_distance_grad


class Agent(Node): 
    def __init__(self):
        super().__init__(
                        "agent", 
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True) 
        
        
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.cc = self.get_parameter("cc").value
        self.beta = self.get_parameter("beta").value
        self.delta = self.get_parameter("delta").value
        self.gamma = self.get_parameter("gamma").value
        self.MAXITERS = self.get_parameter("MAXITERS").value
        self.num_agents = self.get_parameter("NN").value
        self.alpha = self.get_parameter("alpha").value
        self.enable_collision_avoidance = self.get_parameter("Collision_Avoidance").value

        self.get_logger().info(f"Collision Avoidance: {self.enable_collision_avoidance}")

        self.AA = self.get_parameter("AA").value
        AA_shape = self.get_parameter("AA_shape").value

        self.AA = np.array(self.AA).reshape(AA_shape)
        print(f"AA: {self.AA}")

        self.z_i = np.array(self.get_parameter("zzero").value)
        z_i_shape = self.get_parameter("zzero_shape").value

        self.z_i = np.array(self.z_i).reshape(z_i_shape)
        print(f"z_i: {self.z_i}")

        self.r_i = np.array(self.get_parameter("r_i").value)
        r_i_shape = self.get_parameter("r_i_shape").value

        self.r_i = np.array(self.r_i).reshape(r_i_shape)
        print(f"r_i: {self.r_i}")

        self.dd = self.get_parameter("d").value

                
        self.t = 0
        

        self.sigma_z = np.zeros((self.MAXITERS, self.dd))
        self.cost = np.zeros((self.MAXITERS))
        self.grad_norm = np.zeros((self.MAXITERS))
        self.grad = np.zeros((self.MAXITERS, self.num_agents,self.dd))

        self.s_i_error = np.zeros((self.MAXITERS, self.num_agents,self.dd))
        self.v_i_error = np.zeros((self.MAXITERS, self.num_agents, self.dd))

        self.s_i_error_norm = np.zeros((self.MAXITERS, self.num_agents))
        self.v_i_error_norm = np.zeros((self.MAXITERS, self.num_agents))



        for j in range(self.num_agents):
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)


        self.received_data = {j:[] for j in range(self.num_agents)} 

        self.sigma_pub = self.create_publisher(Marker, f"/sigma", 10)

        # here we put a sleep to make sure all agents are created before starting the algorithm
        sleep(2)

    def publish_sigma_pos(self,sigma_z):

        targetmarker = Marker()

        targetmarker.id = self.num_agents + 1 
        targetmarker.header.frame_id = "world"
        targetmarker.ns = "sigma"
        targetmarker.header.stamp = self.get_clock().now().to_msg()
        targetmarker.type = Marker.SPHERE
        targetmarker.pose.position.x = sigma_z[0]
        targetmarker.pose.position.y = sigma_z[1]
        targetmarker.action = Marker.ADD
        targetmarker.scale.x = 0.1
        targetmarker.scale.y = 0.1
        targetmarker.scale.z = 0.1
        targetmarker.color.r = 0.0
        targetmarker.color.g = 1.0
        targetmarker.color.b = 0.0
        targetmarker.color.a = 1.0

        self.sigma_pub.publish(targetmarker)

    # This function is called every time a message is received from a neighbor 
    def listener_callback(self, msg):

        # The message split is into agent id and data 
        j = int(msg.data[0]) 
        msg_j = list(msg.data[1:]) 

        self.received_data[j].append(msg_j) 


        # check if all messages have been received from all agents, if yes, then call timer_callback to perform the updates
        if all(len(self.received_data[j])>0 for j in range(self.num_agents)):
            
            self.timer_callback()

        return None


    # this function is only called when the agent has received all messages from all agents, at time t
    def timer_callback(self): 

        msg = MsgFloat()

       

        # This condition checks if all messages have been received from all agents till time t-1. 
        # It accesses the neighbor [j], and checks the first element of the first element of the list, which is the current time/iteration
        if self.t < self.MAXITERS: 
            all_received = all(
                self.t == self.received_data[j][0][0] for j in range(self.num_agents)
            )
            
            if all_received:

                ## data extraction
                z_agents = np.zeros((self.num_agents, self.dd))
                s_agents = np.zeros((self.num_agents, self.dd))
                v_agents = np.zeros((self.num_agents, self.dd))
                for j in range(self.num_agents):
                    data = self.received_data[j].pop(0)
                    s_agents[j] = np.array(data[1:3])
                    v_agents[j] = np.array(data[3:5])
                    z_agents[j] = np.array(data[5:])
                
                self.sigma_z[self.t] = sigma(self.cc, z_agents)
                
                
                ## cost and gradient computation
                for j in range(self.num_agents):
                    self.cost[self.t] += cost_func(
                        z_agents[j], self.r_i[j], self.sigma_z[self.t], self.r_0, self.gamma, self.beta, self.delta
                    ) 
                    for i in range(self.num_agents):
                        if i != j and self.AA[j,i] != 0:
                            self.cost[self.t] += inter_agent_distance_cost(z_agents[j], z_agents[i],self.enable_collision_avoidance)

                    self.grad[self.t, j] = grad_1_cost(
                        z_agents[j], self.r_i[j], self.sigma_z[self.t], self.gamma, self.delta
                    ) + (1/(self.num_agents)*grad_phi(self.cc, z_agents[j])@grad_2_cost(z_agents[j], self.sigma_z[self.t], self.r_0, self.beta, self.delta))
                    for i in range(self.num_agents):
                        if i != j and self.AA[j,i] != 0:
                            self.grad[self.t, j] += inter_agent_distance_grad(z_agents[j], z_agents[i],self.enable_collision_avoidance)
                
                    ## error computation
                    self.s_i_error[self.t,j] = s_agents[j] - self.sigma_z[self.t]
                    self.v_i_error[self.t,j] = v_agents[j] - grad_2_cost(z_agents[j], self.sigma_z[self.t], self.r_0, self.beta, self.delta)

                    self.s_i_error_norm[self.t,j] = np.linalg.norm(self.s_i_error[self.t,j])
                    self.v_i_error_norm[self.t,j] = np.linalg.norm(self.v_i_error[self.t,j])
                
                
                
                self.grad_norm[self.t] = np.linalg.norm(self.grad[self.t])




                self.publish_sigma_pos(self.sigma_z[self.t])
                if self.t%100==0:
                    self.get_logger().info(f"plotter is at iter {self.t}")


            #start plotting the data at the end of the iterations
            if self.t >= self.MAXITERS-1:
                self.get_logger().info(f"plotter is plotting at iter {self.t}")
                self.plot_Result(self.cost, self.grad_norm, self.s_i_error_norm, self.v_i_error_norm)
                exit()
            
            self.t += 1
            sleep(0.1)
    

    def plot_Result(self, cost, grad_norm, s_i_error, v_i_error):

        MAXITERS = len(cost)

        fig, ax = plt.subplots(1, 2)
        plt.suptitle(f'Agents results')

        ax[0].semilogy(np.arange(MAXITERS-1), cost[:MAXITERS-1])
        ax[0].grid()
        ax[0].set_title(f'Cost')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel(f'Cost')

        ax[1].semilogy(np.arange(MAXITERS-1), grad_norm[:MAXITERS-1])
        ax[1].grid()
        ax[1].set_title(f'Norm of the gradient')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel(rf'Gradient norm')

        fig, ax = plt.subplots(1, 2)
        plt.suptitle(f'Consensus error Norms')
        for j in range(self.num_agents):
            ax[0].plot(np.arange(MAXITERS-1), s_i_error[:MAXITERS-1,j], label=f'agent {j}')
            ax[1].plot(np.arange(MAXITERS-1), v_i_error[:MAXITERS-1,j], label=f'agent {j}')

        ax[0].grid()
        ax[0].set_title(f'Consensus error for sigma')
        ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel(f'error')

        
        ax[1].grid()
        ax[1].set_title(f'Consensus error for nabla 2')
        ax[1].legend()
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel(f'error')

        plt.show()


def main():
    rclpy.init() 

    agent = Agent()
    sleep(1)

    rclpy.spin(agent) 
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

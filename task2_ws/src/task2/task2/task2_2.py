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
from task2.functions import phi, grad_phi, sigma, cost_func, grad_1_cost, grad_2_cost


class Agent(Node): 
    def __init__(self):
        super().__init__(
                        "agent", 
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True) 
        
        self.agent_id = self.get_parameter("id").value 
        self.weights = np.array(self.get_parameter("A_ii").value)
        self.neighbors = self.get_parameter("N_ii").value
        self.z_i = np.array(self.get_parameter("zzero").value)
        self.r_i = np.array(self.get_parameter("r_i").value)
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.cc = self.get_parameter("cc").value
        self.beta = self.get_parameter("beta").value
        self.delta = self.get_parameter("delta").value
        self.gamma = self.get_parameter("gamma").value
        self.MAXITERS = self.get_parameter("MAXITERS").value
        self.alpha = self.get_parameter("alpha").value



        self.get_logger().info(f"Agent: {self.agent_id}")
        self.get_logger().info(f"Agent {self.agent_id} neighbors: {self.neighbors}")
        self.get_logger().info(f"Agent {self.agent_id} r_i: {self.r_i}")
        self.get_logger().info(f"zzero: {self.z_i}")

        

        self.t = 0
        

        #tracker terms for the algorithm
        self.s_i = np.zeros(2)
        self.v_i = np.zeros(2)

        # cost and gradient for plotting later
        self.cost = np.zeros(0)
        self.grad_norm = np.zeros(0)

        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)

        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)                 # topic which publishes agents data e.g states
        self.agent_pos_pub = self.create_publisher(Marker, f"/agent_pos_{self.agent_id}", 10)       # topic which publishes agents position in RVIZ
        self.target_pos_pub = self.create_publisher(Marker, f"/target_pos_{self.agent_id}", 10)      # topic which publishes targets position in RVIZ 
        self.central_target_pos_pub = self.create_publisher(Marker, f"/central_targ", 10)        # topic which publishes central r_i position in RVIZ


        self.received_data = {j:[] for j in self.neighbors} 

        # publish the targets and central r_i in RVIZ when the agent node is created
        self.publish_target_pos()
        self.publish_central_target_pos()

        # here we put a sleep to make sure all agents are created before starting the algorithm
        sleep(2)

        # we call the function to publish the first message which is just the initial states of the agent at t=0
        self.initial_states()


    # This function is called every time a message is received from a neighbor 
    def listener_callback(self, msg):

        # The message split is into agent id and data 
        j = int(msg.data[0]) 
        msg_j = list(msg.data[1:]) 

        self.received_data[j].append(msg_j) 

        self.get_logger().info(f"recived_data[{j}]: {self.received_data[j]}")

        # check if all messages have been received from all neighbors, if yes, then call timer_callback to perform the updates
        if all(len(self.received_data[j])>0 for j in self.neighbors):
            self.timer_callback()

        return None

    
    # This function is called at time t=0 to publish the initial state of the agent
    # and it is only called once at the beginning of the algorithm, by each agent
    def initial_states(self):

        msg = MsgFloat()
        NN=len(self.weights)

        #compute the initial states of si_0 and vi_0 using zi_0
        self.s_i = phi(self.cc, self.z_i)
        self.v_i = grad_2_cost(self.z_i, self.s_i, self.r_0, self.beta, self.delta)

        cost_kk, nabla_1_l = cost_func(self.z_i, self.r_i, self.s_i, self.r_0, self.gamma, self.beta, self.delta), grad_1_cost(self.z_i, self.r_i, self.s_i, self.gamma, self.delta)
        gradient_phi = grad_phi(self.cc, self.z_i)

        norm_grad_k = np.linalg.norm(nabla_1_l + 1/NN*gradient_phi  @ self.v_i)

        # store cost and gradient values for plotting later
        self.cost = np.append(self.cost, cost_kk)
        self.grad_norm = np.append(self.grad_norm, norm_grad_k)

        self.get_logger().info(f"agent {self.agent_id} at iter {self.t}: s is {self.s_i}, v is {self.v_i}  ")

        # using the tracker terms, we create a message to publish to the neighbors
        msg.data = [float(self.agent_id), float(self.t), float(self.s_i[0]), float(self.s_i[1]), float(self.v_i[0]), float(self.v_i[1]), float(self.z_i[0]), float(self.z_i[1])]
        self.publisher.publish(msg)

        # publish the agent's position in RVIZ
        self.publish_agent_trackers()

        # publist the data for plotting

       
        
        # after the computations for t=0 have been done, we increment t by 1 and start the algorithm
        self.t += 1

    # this function is only called when the agent has received all messages from all neighbors, at time t
    def timer_callback(self): 

        msg = MsgFloat()

        

        if self.t > 0: 

            # This condition checks if all messages have been received from all neighbors till time t-1. 
            # It accesses the neighbor [j], and checks the first element of the first element of the list, which is the current time/iteration
            all_received = all(
                self.t-1 == self.received_data[j][0][0] for j in self.neighbors
            )

            if all_received:
                
                cost_kk, nabla_1_l = cost_func(self.z_i, self.r_i, self.s_i, self.r_0, self.gamma, self.beta, self.delta), grad_1_cost(self.z_i, self.r_i, self.s_i, self.gamma, self.delta)

                gradient_phi = grad_phi(self.cc, self.z_i)

                NN=len(self.weights)
                z_k1 = self.z_i - self.alpha * ( nabla_1_l + gradient_phi  @ self.v_i)

                norm_grad_k = np.linalg.norm(nabla_1_l + 1/NN*gradient_phi  @ self.v_i)

                self.cost = np.append(self.cost, cost_kk)
                self.grad_norm = np.append(self.grad_norm, norm_grad_k)

                sk_1 = np.zeros(2)
                vk_1 = np.zeros(2)

                # computing innovation term for sk_1
                phi_kk1 = phi(self.cc,z_k1)
                phi_kk = phi(self.cc,self.z_i)

                sk_1 = (phi_kk1 - phi_kk)

                for j in self.neighbors:                 
                    data_from_j = self.received_data[j].pop(0) 

                    s_j = np.array(data_from_j[1:3]) 
                    v_j = np.array(data_from_j[3:5]) 

                    sk_1 += self.weights[j] * s_j
                    vk_1 += self.weights[j] * v_j
                
                # computing innovation term for vk_1
                nabla_2_l_kk  = grad_2_cost(self.z_i, self.s_i, self.r_0, self.beta, self.delta)
                nabla_2_l_kk1 = grad_2_cost(z_k1, sk_1, self.r_0, self.beta, self.delta)

                vk_1 += nabla_2_l_kk1 - nabla_2_l_kk 

                # update the agent's states and tracker terms for the next iteration
                self.z_i = z_k1 
                self.s_i = sk_1
                self.v_i = vk_1  

                # create message to publish using the updated states and tracker terms
                msg.data = [float(self.agent_id), float(self.t), float(self.s_i[0]), float(self.s_i[1]), float(self.v_i[0]), float(self.v_i[1]), float(self.z_i[0]), float(self.z_i[1])]
                self.publisher.publish(msg) 

                

                # print the data for the agent at time t at each 10th iteration
                if self.t%100==0:
                    self.get_logger().info(f"agent {self.agent_id} at iter {self.t}: z is {self.z_i}, s is {self.s_i}, v is {self.v_i} ")

                # publish data to RVIZ if MAXITERS is not achieved
                if self.t < self.MAXITERS:
                    self.publish_agent_trackers()

                # otherwise, start plotting the data
                else:
                    self.get_logger().info(f"agent {self.agent_id} at iter {self.t}: exiting... ")
                    # self.get_logger().info(f"agent {self.agent_id} at iter {self.t}: plotting... ")
                    # self.plot_Result(self.cost, self.grad_norm, self.agent_id)
                    exit()
                
                self.t += 1
                sleep(0.1)
    
    '''
        The following functions are used for visualization in RVIZ
        The functions for static targets are only called once
        The functions for the agent's position and barycenter are called at each iteration
    '''

    # this function publishes the r_i of the agent, i.e. the intruders, in RVIZ
    def publish_target_pos(self):

        targetmarker = Marker()

        targetmarker.id = self.agent_id
        targetmarker.header.frame_id = "world"
        targetmarker.ns = f"target_{self.agent_id}"
        targetmarker.header.stamp = self.get_clock().now().to_msg()
        targetmarker.type = Marker.SPHERE
        targetmarker.pose.position.x = self.r_i[0]
        targetmarker.pose.position.y = self.r_i[1]
        targetmarker.action = Marker.ADD
        targetmarker.scale.x = 0.1
        targetmarker.scale.y = 0.1
        targetmarker.scale.z = 0.1
        targetmarker.color.r = 1.0
        targetmarker.color.g = 0.0
        targetmarker.color.b = 0.0
        targetmarker.color.a = 1.0

        self.target_pos_pub.publish(targetmarker)

    # this function publishes the central r_i in RVIZ, i.e. the r_i for the barycenter
    def publish_central_target_pos(self):

        targetmarker = Marker()

        targetmarker.id = self.agent_id
        targetmarker.header.frame_id = "world"
        targetmarker.ns = f"target_{self.agent_id}"
        targetmarker.header.stamp = self.get_clock().now().to_msg()
        targetmarker.type = Marker.SPHERE
        targetmarker.pose.position.x = self.r_0[0]
        targetmarker.pose.position.y = self.r_0[1]
        targetmarker.action = Marker.ADD
        targetmarker.scale.x = 0.1
        targetmarker.scale.y = 0.1
        targetmarker.scale.z = 0.1
        targetmarker.color.r = 1.0
        targetmarker.color.g = 1.0
        targetmarker.color.b = 0.0
        targetmarker.color.a = 1.0

        self.central_target_pos_pub.publish(targetmarker)

    # this function publishes the agent's position in RVIZ
    def publish_agent_trackers(self):

        agentmarker = Marker()
        
        agentmarker.id = self.agent_id
        agentmarker.header.frame_id = "world"
        agentmarker.ns = f"agent_{self.agent_id}"
        agentmarker.header.stamp = self.get_clock().now().to_msg()
        agentmarker.type = Marker.SPHERE
        agentmarker.pose.position.x = self.z_i[0]
        agentmarker.pose.position.y = self.z_i[1]
        agentmarker.action = Marker.ADD
        agentmarker.scale.x = 0.1
        agentmarker.scale.y = 0.1
        agentmarker.scale.z = 0.1
        agentmarker.color.r = 0.0
        agentmarker.color.g = 0.0
        agentmarker.color.b = 1.0
        agentmarker.color.a = 1.0

        self.agent_pos_pub.publish(agentmarker)
        




def main():
    rclpy.init() 

    agent = Agent()
    sleep(1)

    rclpy.spin(agent) 
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

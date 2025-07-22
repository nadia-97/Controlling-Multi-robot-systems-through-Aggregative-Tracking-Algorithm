# Distributed-Autonomous-Systerms

README FOR TASK 2

Below you can find the information on how to run each sub task:

__________________________________________________________________________________________________________________________________________

TASK 2.1: Aggregative Tracking in Python

- Open file task_2_1

- You can set the following parameters as you wish from the file directly:
	- NN (number of agents)
	- alpha (learning rate)
	- MAXITERS (Maximum Iterations for the update steps)
	- r_0 (Position of the Central Target)
	- MOVE_R_I (Set this Boolean to True to have moving intruders)
	- cc (Weights for a weighted average positions (sigma))
	- beta (to pursue r_0 through barycenter
	- delta (to make agents pursue the barycenter)
	- gamma (to make agents pursue the intruders)

- Now run the python file

__________________________________________________________________________________________________________________________________________

ROS2 WORKSPACE SET UP

- Copy the task2_ws folder in your ros2 environment
- open terminal and make sure you are in the task2_ws folder
- copy the following command to build the package: colcon build
- then, copy the following command: . install/setup.bash

Now you are ready to run tasks 2.2 and 2.3
__________________________________________________________________________________________________________________________________________

TASK 2.2: Aggregative Tracking in ROS 2

- You can change the following parameters from the task2_launch.launch.py but make sure the run the commands mentioned above after making the changes. 
	- NN (number of agents)
	- alpha (learning rate)
	- MAXITERS (Maximum Iterations for the update steps)
	- r_0 (Position of the Central Target)
	- cc (Weights for a weighted average positions (sigma))
	- beta (to pursue r_0 through barycenter
	- delta (to make agents pursue the barycenter)
	- gamma (to make agents pursue the intruders)

- Now, follow the steps below:

Step 1: Copy the following command in the terminal to first launch rviz2 with a set configuration: ros2 run rviz2 rviz2 -d <Copy PATH of the config.rviz file from the task2_ws> 

An example of this is shown below: 
EXAMPLE: ros2 run rviz2 rviz2 -d /home/nadia/task2_ws/task2_ws/src/task2/config/config.rviz

Step 2: Now in a separate terminal, copy the following command to launch the agents & the plotter nodes: ros2 launch task2 task2_2_launch.launch.py
__________________________________________________________________________________________________________________________________________

TASK 2.3: Moving in a corridor via Projected Aggregative tracking algorithm

- You can change the following parameters from the task2_launch.launch.py but make sure the run the commands mentioned above after making the changes. 
	- NN (number of agents)
	- alpha (learning rate)
	- MAXITERS (Maximum Iterations for the update steps)
	- cc (Weights for a weighted average positions (sigma))
	- beta (to pursue r_0 through barycenter
	- delta (to make agents pursue the barycenter)
	- gamma (to make agents pursue the intruders)
	- Enable_Collision_Avoidance (Set this to true to prevent collisions between agents)
	- zeta (this is the weight given to the difference between the actual and projected positions of the agents)
	- inter_agent_safe_distance (defines the safe distance to be kept among the agents)
	- corridor_safe_distance (defines the safe distance from the corridor)

- Now, follow the steps below:

Step 1: Copy the following command in the terminal to first launch rviz2 with a set configuration: ros2 run rviz2 -d <Copy PATH of the config.rviz file from the task2_ws> 

An example of this is shown below: 
EXAMPLE: ros2 run rviz2 rviz2 -d /home/nadia/task2_ws/task2_ws/src/task2/config/config.rviz

Step 2: Now in a separate terminal, copy the following command to launch the agents & the plotter nodes: ros2 launch task2 task2_3_launch.launch.py





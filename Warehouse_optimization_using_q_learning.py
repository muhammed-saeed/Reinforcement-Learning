
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:56:48 2020

@author: muhammed
"""

'''
###########AI To OPtimize the Business and reach the goal of the system################
#takes as input the starting location for where the robot has started, also hte location for where the robot has started the motion
#return hte path composed of the location those we have to pass through from the starting point to the ending point
#the starting location is always the row of the Q-value and  the final location is the location corresponding to the heightest Q-value
#One can select all the above code and then choose the actions those allows you to Get the maximum Q-value
#always the robot is going to choose the location corresponding to the heights Q-values statring from the start state and going to hte target  locatio n
# the row is the row corresponding to the starting location
# each action takes us to state, and playable actions are those which allows us to get from state to another state all the other actions are going to be Zero
'''





import numpy as np

gamma = 0.75
alpha = 0.9

#part define the environment

#define the state
location_to_state = {'A':0,
                     'B':1,'C':2,
                     'D':3,'E':4,
                     'F':5,'G':6,
                     'H':7,'I':8,
                     'J':9,'K':10,
                     'L':11}
#change from the keys [a,b,c,d,e,f,g,h] --> into states [1,2,3,4,5,6]

#define the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]
#action 0 means the action to go to state 0, and action 1 is the action to go to state one and so on

#define the rewards
#the reward function is a function in both the state and the actions
#the reward function can be as a function of both the states which are rows and the actions which are columns
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])
#note we may make hte location we want to reach manunaling by change the value of location G,G to 1000 so hte agent by default will find the shortest path to reach there


#PART 2--building the AI solution
'''
1-make initialy all the states and actions to zero
2- determine the current_state by choosing the action randomy
3- Play an Action and see which next state you reached, first we need to determine the actions and call them playable actions which means the actions those have corresponding non-zero rewards and from there play random action
4-
5-
'''

Q = np.array(np.zeros([12,12]))
#since for each state we can do all the 12 actions then we have table with size state x actions
#first make all the Q(s,a) values to be zero and change it as you progress

for i in range(1000):
 #we want to repeat this process 1000 times
    
    current_state = np.random.randint(0, 12)
    #note the upper-bound in python is excluded, so here we go from 0 to 11
    
    playable_actions = []
    # and we are going to append to the playable_actions the index of the state in which the reward isnot zero
    
    for j in range(12):
        if R[current_state,j] > 0:
            playable_actions.append(j)
    
    next_state = np.random.choice(playable_actions)
    #note the playable_actions will indeed leads us to the next state
    #the action_we_played is the same as the next_state
    
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    

state_to_location = {state: location for location, state in location_to_state.items()}
 #this is the mapping function that maps from state in numbers into locations in letters   
    
#PART 3--going to the production
def route(starting_location, ending_location):
#takes as input the starting location for where the robot has started, also hte location for where the robot has started the motion
#return hte path composed of the location those we have to pass through from the starting point to the ending point
#the starting location is always the row of the Q-value and  the final location is the location corresponding to the heightest Q-value
#One can select all the above code and then choose the actions those allows you to Get the maximum Q-value
#always the robot is going to choose the location corresponding to the heights Q-values statring from the start state and going to hte target  locatio n
#location is letter like(A, B,C,D, ) and the state are values like (0,1,2,3,4,5,)
    
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12,12]))
#since for each state we can do all the 12 actions then we have table with size state x actions
#first make all the Q(s,a) values to be zero and change it as you progress

    for i in range(1000):
     #we want to repeat this process 1000 times
        
        current_state = np.random.randint(0, 12)
        #note the upper-bound in python is excluded, so here we go from 0 to 11
        
        playable_actions = []
        # and we are going to append to the playable_actions the index of the state in which the reward isnot zero
        
        for j in range(12):
            if R_new[current_state,j] > 0:
                playable_actions.append(j)
        
        next_state = np.random.choice(playable_actions)
        #note the playable_actions will indeed leads us to the next state
        #the action_we_played is the same as the next_state
        
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
        
    
    state_to_location = {state: location for location, state in location_to_state.items()}
     #this is the mapping function that maps from state in numbers into locations in letters   
        
    route = [starting_location]
    next_location = starting_location
    #this is the location that corresponds to the heights Q-value statring from the first location
    while (next_location != ending_location):
        starting_location = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_location,])
        #make an inverse mapping to convert the index of the state to the corresponding letter
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    
    return route

def best_route(starting_location, intermediate_location, ending_location):
    return route(starting_location, intermediate_location) + route(intermediate_location, ending_location)

print('Route: ')
best_route('E','F' ,'D')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_actions
        self.fc1 = nn.Linear(input_size, 30)

        self.fc2 = nn.Linear(30, nb_actions)

    def forward(self, state):
        #takes the input state and return as Output the Q-Functions of the state
        x = F.relu(self.fc1(state))
        #we use the relu Activation


        q_values = self.fc2(x)
        return q_values

#Experience Replay
#note this is the same as the time-series
#so our serues wont be t and t-1 but will be some transitions of hte past and this allows us to do the best
#and we sample which means takse random samples from this

class ReplayMemory(object):
    def __init__(self, capacity):
          self.capacity = capacity
          self.memory = []
          #here we determine the length of the experiecnce replay memory

    def push(self,event):
        #push the elements to the experience memory and also makes sure the size of the list is in the size of the capacity
        self.memory.append(event)
        if len(self.memory) > self.capacity :
          del self.memory[0]
          #(state, next_state, action, reward) <-- each
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

        #what does the zip do?
        ## its a reshape function ((1,2,3,),(4,5,6)) --> zip function ((1,4),(2,5),(3,6))
        ##(state1, action1, reward1),(state2, action2, reward2) --> (state1, state2),(action1, action2),(reward1 , reward2)
        ##since we pass this to  pytorch Variable and pytorch variable contains both a tensor and a variable
        #and for pytorch to work we have to pass both the tensor f
        #random.sample(takes the population which is in our case is the memory and also takes the batch_size)

        #we apply this lambda  function on all the samples and get the desired output


#implementing the Deep-Q-Learning Model

class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        #the idea of hte softmax we are trying to get the best action to play while in the same time explores other actions

        #note state myst be as torch tensor
        #note the temperature parameter allows the network to become more sure about the result
        #temperature is about the certainity we must take care about with each action
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = torch.multinomial(probs, 1)
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #llec 39 needs revisit and get somestuff clearer
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #but we are interested the actions those choosen and played by the agent
        #note the gather(1 & batch_action) will return teh best action to take
        #batch_state has the fack dimension corresponding to the batch but we donot have that batch_serise that correspond to the agent
        #now we kill the fake dimension, since we get the desired output

        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #note since the next state q_value is taken with respect to all the possible actions then we use detach method
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        #we must reinitialize the optimizer at each iteration of the loop
        td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_signal):
        # this function updates every thing when the AI select new action
        #so the next_state becomes the new state and aslo that each action and reward must be added to the batches

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

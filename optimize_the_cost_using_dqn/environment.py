import numpy as np

class Environment(object):
    #The idea is to use Deep Reinforcement Learning Techniques to reduce the cost of datacenter, measuring the level of humidity, number of users and many other features
    #then choose the best action accordingly
    #introducing and installing all the parameters and variables
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data  = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    #making a method that update the Envrionment right after the AI agent plays an action
    def update_env(self, direction, energy_ai, month):
        #the energy spent by the AI that changes the server's temperature from Ti to Ti+1 within 1 unit time
        #can be approximated again
        energy_noai = 0
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai =  self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]

        elif(self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]

        self.reward = energy_noai - energy_ai

        self.reward = 1e-3 * self.reward

        #update hte atmospheric_temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]

        #update the number of users
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users

        #update the rate of the data 
        self.current_rate_data += np.random.randint(-self.max_rate_data, self.max_rate_data)
        if(self.current_rate_data >  self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif(self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data

        #calculate the change of the intrinsic_temperature
        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        #note the above line of code contains both the new number of users and also the new
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif(direction == 1):
            delta_temperature_ai = energy_ai

        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai

        self.temperature_noai += delta_intrinsic_temperature

        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai

        elif(self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]

        #update the total energy spent by the AI
        self.total_energy_ai += energy_ai
        #update the total energy used by the server when there is no ai
        self.total_energy_noai += energy_noai

       
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) /(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])

        return next_state, self.reward, self.game_over

    #the reset function as its name indicates it resets the envrionment
    def reset(self,new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    #observe method sense environment - MDP - and get the required obseravations which form the state
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) /(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])

        return current_state, self.reward, self.game_over
    #making method that gives us at any current state, the last reward

"""
This code has been adopted from the github repo EthanMacdonald/h-DQN
I am commenting this code so that people can understand it more easily
Starting : This code is based on the paper over here broadly we have 2 neural networks
1. The Meta Polciy Neural Network for Q function
2. The controller level Neural Network which encompasses all the option policies in a single Neural Network, switched by a one hot vector for each option polcy
"""

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop

# Default architecture for the meta controller
default_meta_layers = [Dense] * 5
default_meta_inits = ['lecun_uniform', 'lecun_uniform', 'lecun_uniform', 'lecun_uniform', 'lecun_uniform']
default_meta_nodes = [6, 30, 30, 30, 6] # The state is 6 long and there are 6 options
default_meta_activations = ['relu', 'relu', 'relu', 'relu', 'relu']
default_meta_loss = "mean_squared_error"
default_meta_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_meta_n_samples = 1000 # size of batch to sample from meta replay memeory
default_meta_epsilon = 1.0

# Default architectures for the lower level controller/actor
default_layers = [Dense] * 5
default_inits = ['lecun_uniform'] * 5
default_nodes = [12, 30, 30, 30, 2] # The state is concat of the 6 from meta polcy and the 6 represnting the one hot vector for each option, and there are 2 primitive actions
default_activations = ['relu'] * 5
default_loss = "mean_squared_error"
default_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_n_samples = 1000 # size the mini batch to sample from replay memory
default_gamma = 0.975
default_epsilon = 1.0
default_actor_epsilon = [1.0]*6 # We need to maintain a separate epsilon for each option
default_tau = 0.001

class hDQN:
    def __init__(self, meta_layers=default_meta_layers, meta_inits=default_meta_inits,
                meta_nodes=default_meta_nodes, meta_activations=default_meta_activations,
                meta_loss=default_meta_loss, meta_optimizer=default_meta_optimizer,
                layers=default_layers, inits=default_inits, nodes=default_nodes,
                activations=default_activations, loss=default_loss,
                optimizer=default_optimizer, n_samples=default_n_samples,
                meta_n_samples=default_meta_n_samples, gamma=default_gamma,
                meta_epsilon=default_meta_epsilon, epsilon=default_epsilon, actor_epsilon = default_actor_epsilon, tau = default_tau):
        self.meta_layers = meta_layers
        self.meta_inits = meta_inits
        self.meta_nodes = meta_nodes
        self.meta_activations = meta_activations
        self.meta_loss = meta_loss
        self.meta_optimizer = meta_optimizer
        self.layers = layers
        self.inits = inits
        self.nodes = nodes
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        self.meta_controller = self.meta_controller()
        self.target_meta_controller = self.target_meta_controller()
        self.actor = self.actor()
        self.target_actor = self.target_actor()
        self.goal_selected = np.ones(6)
        self.goal_success = np.zeros(6)
        self.meta_epsilon = meta_epsilon
        self.actor_epsilon = actor_epsilon
        self.n_samples = n_samples
        self.meta_n_samples = meta_n_samples
        self.gamma = gamma
        self.target_tau = tau
        self.memory = []
        self.meta_memory = []

    def meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](self.meta_nodes[0], init=self.meta_inits[0], input_shape=(self.meta_nodes[0],)))
        meta.add(Activation(self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, init=init, input_shape=(node,)))
            meta.add(Activation(activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta
    
    def target_meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](self.meta_nodes[0], init=self.meta_inits[0], input_shape=(self.meta_nodes[0],)))
        meta.add(Activation(self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, init=init, input_shape=(node,)))
            meta.add(Activation(activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta


    def actor(self):
        actor = Sequential()
        actor.add(self.layers[0](self.nodes[0], init=self.inits[0], input_shape=(self.nodes[0],)))
        actor.add(Activation(self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            print(node)
            actor.add(layer(node, init=init, input_shape=(node,)))
            actor.add(Activation(activation))
        actor.compile(loss=self.loss, optimizer=self.optimizer)
        return actor
    
    def target_actor(self):
        actor = Sequential()
        actor.add(self.layers[0](self.nodes[0], init=self.inits[0], input_shape=(self.nodes[0],)))
        actor.add(Activation(self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            print(node)
            actor.add(layer(node, init=init, input_shape=(node,)))
            actor.add(Activation(activation))
        actor.compile(loss=self.loss, optimizer=self.optimizer)
        return actor

    def select_move(self, state, goal, goal_value):
        vector = np.concatenate([state, goal], axis=1)
        if random.random() < self.actor_epsilon[goal_value-1]:
            return np.argmax(self.actor.predict(vector, verbose=0))
        return random.choice([0,1])

    def select_goal(self, state):
        if self.meta_epsilon < random.random():
            pred = self.meta_controller.predict(state, verbose=0)
            print("pred shape: " + str(pred.shape))
            return np.argmax(pred)+1
        print("Exploring")
        return random.choice([1,2,3,4,5,6])

    def criticize(self, goal, next_state):
        return 1.0 if goal == next_state else 0.0

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
            if len(self.meta_memory) > 1000000:
                self.meta_memory = self.meta_memory[-100:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 1000000:
                self.memory = self.memory[-1000000:]

    def _update(self):
        exps = [random.choice(self.memory) for _ in range(self.n_samples)] # sample n_samples from the controller memory
        state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps])) # for each experince exp contains the state, the goal, action, reward, next state and the same goal, hence we need to make the state vector by concat the s and g vectors
        next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=1) for exp in exps])) # the same and the squeeze operator converts the 2 d array into a 1 d Array where ever there is single dimension
        try:
            reward_vectors = self.actor.predict(state_vectors, verbose=0) # accumulate the Q(s,a) functions for each action
        except Exception as e:
            state_vectors = np.expand_dims(state_vectors, axis=0)
            reward_vectors = self.actor.predict(state_vectors, verbose=0) # accumulate the Q(s,a) functions for each action
        
        try:
            next_state_reward_vectors = self.target_actor.predict(next_state_vectors, verbose=0) # Accumulate the Q(s',a') for each action in the s' state
        except Exception as e:
            next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
            next_state_reward_vectors = self.target_actor.predict(next_state_vectors, verbose=0)
        # Also get the Q values from the target network
        
        for i, exp in enumerate(exps):
            reward_vectors[i][exp.action] = exp.reward # Put the reward for action if the next state was terminal as the reward attained for tha state, i.e. this will only happen if after this state the epsiode or controller terminates
            if not exp.done:
                reward_vectors[i][exp.action] += self.gamma * max(next_state_reward_vectors[i])
        reward_vectors = np.asarray(reward_vectors)
        self.actor.fit(state_vectors, reward_vectors, verbose=0)
        
        #Update target network
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.target_tau * actor_weights[i] + (1 - self.target_tau) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            exps = [random.choice(self.meta_memory) for _ in range(self.meta_n_samples)]
            state_vectors = np.squeeze(np.asarray([exp.state for exp in exps]))
            next_state_vectors = np.squeeze(np.asarray([exp.next_state for exp in exps]))
            try:
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            
            try:
                next_state_reward_vectors = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            except Exception as e:
                next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
                next_state_reward_vectors = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            
            for i, exp in enumerate(exps):
                reward_vectors[i][np.argmax(exp.goal)] = exp.reward
                if not exp.done:
                    reward_vectors[i][np.argmax(exp.goal)] += self.gamma * max(next_state_reward_vectors[i])
            self.meta_controller.fit(state_vectors, reward_vectors, verbose=0)
            
            #Update target network
            meta_weights = self.meta_controller.get_weights()
            meta_target_weights = self.target_meta_controller.get_weights()
            for i in range(len(meta_weights)):
                meta_target_weights[i] = self.target_tau * meta_weights[i] + (1 - self.target_tau) * meta_target_weights[i]
            self.target_meta_controller.set_weights(meta_target_weights)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()
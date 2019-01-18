"""
This enviornment will be hardcoded approach for the proposed system
"""


import numpy as np
import random
import utils
import impdicts


class MetaEnv:

    def __init__(self, w1 : float=  1, w2 : float = 8, w3 : float = 13,   intent_space_size : int = 5, slot_space_size : int  =  8, options_space : int = 5, primitive_action_space : int = 20 ):
        """
        Args
        :param intent_space_size: The number of intents in the system
        :param slot_space_size: The number of slot that need to be filled with the probability values
        :param options_space: The number of available options in the meta policy
        :param primitive_action_space:  The number of primitve actions available
        :param current_intents: Array containing the active intents for the current obj
        """
        
        self.intent_space_size = intent_space_size
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space

        self.current_obj_intent =[] # This shows all the intents that have to be served in this object
        self.slot_states = []
        self.current_intent_state= []
        self.current_slot_state = []
        self.current_intent_no = 0
        self.no_intents = 0 # the number of intents to be served i.e. len(self.current_obj_intent)
        self.reset()
        # self.slot_states # this is the list of all the slot states encountered in runs
        # self.current_slot_state # this is state of the confidence values in the current context
        # self.current_intent_state # This contains the current intent that the env is focusing on (currently a one hot vector later, maybe changed to a composite of multiple intents
        # self.intent_states # collection of the timeline of the intent states encountered



    def reset(self, prob_random_start : float = 0.5) :
        """
        confidence_state, intent_state = env.reset()
        We need to init the object with a set of new intents

        :return: confidence_state : The slot state with confidence  values
        intent_state : The intent state for the current intent
        """
        self.current_obj_intent = []
        self.no_intents = np.random.randint(1,self.intent_space_size + 1) # Produces a number between 1 and 5 for the number of intents
        # get a starting number of intents from 0 to intent_space
        temp_intents = list(range(0, self.intent_space_size))
        for iter in range(self.no_intents):
            indx = np.random.randint(0, len(temp_intents))
            self.current_obj_intent.append(temp_intents[indx])
            del temp_intents[indx]
        # after these steps teh current_onbj_intent contains the intents and there order to be followred in sccheduling intents for the agent
        if random.random() < prob_random_start:
            # do a random init
            self.random_state_init()
        else:
            self.state_init()
        # now we will set the intital intent state of the system and also the buffer to store the intent states
        self.current_intent_no = 0 # Keeps track of the intent number being served
        self.intent_states = np.array([ utils.one_hot( self.current_obj_intent[self.current_intent_no], self.intent_space_size)]) # setting the starting intent
        self.current_intent_state = self.intent_states[-1]
        return [self.current_slot_state, self.current_intent_state]


    def random_state_init(self):
        self.slot_states = np.array([[random.random() for i in range(self.slot_space_size)]])
        self.current_slot_state = self.slot_states[-1]

    def state_init(self):
        self.slot_states = np.zeros((1, self.slot_space_size))  # initliase with zero for all values
        self.current_slot_state = self.slot_states[-1]


    def step(self, action):
        pass

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int):
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward 
        """
        relevants_slots = impdicts.intent2slots[goal]
        


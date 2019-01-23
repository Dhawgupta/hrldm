"""
This enviornment will be hardcoded approach for the proposed system
"""


import numpy as np
import random
import utils
import impdicts
from typing import List, Tuple, Dict
import sys
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
        self.threshold = 0.7
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3 # probably will not be using this
        self.intent_space_size = intent_space_size
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.current_obj_intent =[] # This shows all the intents that have to be served in this object
        self.slot_states = []
        self.current_intent_state= []
        self.current_slot_state = np.array([])
        self.current_intent_no = 0
        self.no_intents = 0 # the number of intents to be served i.e. len(self.current_obj_intent)
        self.goal_iter = [] # The number of iterations done for each subgoal completion
        self.reset()
        self.latest_start_confidence_start = [] # Store the last confidnce state before the start of the option play for the subpolicy
        # self.slot_states # this is the list of all the slot states encountered in runs
        # self.current_slot_state # this is state of the confidence values in the current context
        # self.current_intent_state # This contains the current intent that the env is focusing on (currently a one hot vector later, maybe changed to a composite of multiple intents
        # self.intent_states # collection of the timeline of the intent states encountered



    def reset(self, prob_random_start : float = 0.5): # just trying this return syntax
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


    def step(self, action : int):
        """
        Addressing the monster in the room, this is the holy grail of the whole system
        :param action: The primitive action that is supposed to be taken
        :return: Returns a tuple of [next_confidence_state, intent_state] , intrinsic_reward, goal_reached , done
        Rules and Assumptions
        1. Whenever I recieve a terminating action for a subgoal i.e. the 19 action i will move onto the next intent_state, and also make the goal_reached variable as true
        2. When I am out of processing the next intents, I will move onto making the done variable as true.
        3. Whenever an action comes that is out of the reason of the current intent, we won't modify the state, and we will penalize the aciton by coinsidering the negtative reward from a step that will be consumed in doing that step.
        4. Currently I am also giving the extrinsic reward when taking the terminating action, but we can later remove, thsi , this is done because we need to encourage the agent to somehow learn the terminating action.

        """
        print("Env Step")
        done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state) # copy the state of the current slot state
        current_intent = self.current_obj_intent[self.current_intent_no]
        print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent) :
                # give full reward
                reward = self.w2*self.calculate_external_reward(np.zeros(self.slot_space_size), self.current_slot_state, current_intent)
            else:
                reward = -self.w2*self.calculate_external_reward(self.current_slot_state, np.ones(self.slot_space_size), goal = current_intent)
            self.current_intent_no += 1 # if all the intents in the current object are over
            if self.current_intent_no >= self.no_intents:
                done = True
            else:
                self.current_intent_state = utils.one_hot(self.current_obj_intent[self.current_intent_no, self.intent_space_size])
            goal_reached = True
            return [self.current_slot_state, self.current_intent_state], reward, goal_reached, done

        # if not terminating action
        relevant_actions : List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2*random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot])*0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2*random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2*self.calculate_external_reward(self.current_slot_state, new_state, current_intent) - self.w1 # get the reward for increase in confidecne and the decrrease in iteration
            # Although the calculate external reward is not required as we are already cross checking the things in the first if condition

        self.current_intent_state = np.copy(new_state)
        return [ self.current_intent_state, self.current_intent_state], reward, False, False

    def controller_step(self, goal, action):
        """

        :param goal: The goal the controller is fulfilling to appropriately reward it
        :param action: : The action that the agent is taking pertaining to that goal
        :return: next_confidence_state, reward, goal_completed
        """
        print("Env Controller Step")
        # done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state)  # copy the state of the current slot state
        current_intent = goal
        print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent):
                # give full reward
                reward = self.w2 * self.calculate_external_reward(np.zeros(self.slot_space_size),
                                                                  self.current_slot_state, current_intent)
            else:
                reward = -self.w2 * self.calculate_external_reward(self.current_slot_state,
                                                                   np.ones(self.slot_space_size), goal=current_intent)
            # shifting the below part in the meta step acitons
            # self.current_intent_no += 1  # if all the intents in the current object are over

            goal_reached = True
            return self.current_slot_state, reward, goal_reached,

        # if not terminating action
        relevant_actions: List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot]) * 0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2 * self.calculate_external_reward(self.current_slot_state, new_state,
                                                              current_intent) - self.w1  # get the reward for increase in confidecne and the decrrease in iteration
            # Although the calculate external reward is not required as we are already cross checking the things in the first if condition

        self.current_slot_state = np.copy(new_state)
        return self.current_slot_state, reward, False

    def meta_step_start(self,option):
        """
        This will be responsible for storing the current_confidence-state, to make a comparison with the same in the future
        :param option: Take as intput the option chosed by the meta policy
        :return: return the current confidence state of the dialogue
        """
        # store the current_confidence state
        self.latest_start_confidence_start = np.copy(self.current_slot_state)
        return self.latest_start_confidence_start

    def meta_step_end(self, option) -> Tuple[int, float, bool]  :
        """

        :param option: The option chosen
        :return: return the  next intent, reward, done (still skeptical about returning the confidence state)
        Points :
        Currently the rewards a really simple, they dont penalize for filling other slots that might not be relevant. We can penalize these states in the future by subtracting the extra sltos filled
        """
        done = False
        reward = 0
        current_intent = self.current_obj_intent[self.current_intent_no]
        reward = self.w2*self.calculate_external_reward(np.copy(self.latest_start_confidence_start), self.current_slot_state, current_intent)
        self.current_intent_no += 1  # if all the intents in the current object are over
        new_intent = current_intent
        if self.current_intent_no >= self.no_intents:
            done = True
        else:
            self.current_intent_state = utils.one_hot(
                self.current_obj_intent[self.current_intent_no], self.intent_space_size)
            new_intent = self.current_obj_intent[self.current_intent_no]
        return self.current_intent_state, reward, done

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int) -> float :
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward 
        """
        relevants_slots = impdicts.intent2slots[goal] # this will return the slots to be measured
        # now we will calculate the differen from both
        diff_confidence: float = np.sum(goal_state[relevants_slots] - start_state[relevants_slots])
        # now we can multiply by weights
        return diff_confidence # or we can keep and differnt weight factor for the external agent

    def check_confidence_state(self, goal):
        """
        THis function will check all the slots pertaining to the goal and see if its satisfies the threshold, if it does so , it returns a True value, otherwise it returns a false value.
        :param goal: The goal that its currently checking for
        :return: True/False
        Rules & Points :
        1. I am using the threshold currently to analyze
        """
        slots_to_be_checked = impdicts.intent2slots[goal] # these are the slots
        return all(self.current_slot_state[slots_to_be_checked] > self.threshold)

# prepare and environment to train a separate conteroller
class ControllerEnv:
    def __init__(self, goal = 0, w1 = 1, w2 = 8, w3 = 0, slot_space_size = 8, options_space = 5, primitive_action_space = 20, goal_space_size = 5): # The default goal
        self.goal = goal
        self.threshold = 0.7
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3  # probably will not be using this
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.slot_states = []
        self.current_goal_state = []
        self.current_slot_state = np.array([])
        self.goal_space_size = goal_space_size
        self.goal = goal
        self.goal_iter = []  # The number of iterations done for each subgoal completion
        self.reset(goal = goal)

    def reset(self, goal= 0, state = None, prob_random_start : float = 0.5):
        self.goal = goal

        if random.random() < prob_random_start:
            # do a random init
            self.random_state_init()
        else:
            self.state_init()
        # now we will set the intital intent state of the system and also the buffer to store the intent states
        self.goal = goal  # Keeps track of the intent number being served
        self.goal_state = np.array([utils.one_hot(self.goal,self.goal_space_size)])  # setting the starting intent
        self.current_goal_state = self.goal_state[-1]
        return [self.current_slot_state, self.current_goal_state]

    def random_state_init(self):
        self.slot_states = np.array([[random.random() for i in range(self.slot_space_size)]])
        self.current_slot_state = self.slot_states[-1]

    def state_init(self):
        self.slot_states = np.zeros((1, self.slot_space_size))  # initliase with zero for all values
        self.current_slot_state = self.slot_states[-1]

    def check_confidence_state(self, goal):
        """
        THis function will check all the slots pertaining to the goal and see if its satisfies the threshold, if it does so , it returns a True value, otherwise it returns a false value.
        :param goal: The goal that its currently checking for
        :return: True/False
        Rules & Points :
        1. I am using the threshold currently to analyze
        """
        slots_to_be_checked = impdicts.intent2slots[goal] # these are the slots
        return all(self.current_slot_state[slots_to_be_checked] > self.threshold)

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int) -> float :
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward
        """
        goal = self.goal
        relevants_slots = impdicts.intent2slots[goal] # this will return the slots to be measured
        # now we will calculate the differen from both
        diff_confidence: float = np.sum(goal_state[relevants_slots] - start_state[relevants_slots])
        # now we can multiply by weights
        return diff_confidence # or we can keep and differnt weight factor for the external agent

    def constrain_actions(self, goal = None):
        """
        THis will return
        :param goal:
        :return:
        """
        if goal is None:
            pass
        else:
            pass
        return list(range(self.primitive_action_space))

    def step(self, action):
        print("Env Controller Step")
        # done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state)  # copy the state of the current slot state
        current_intent = self.goal
        print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent):
                # give full reward
                reward = self.w2 * self.calculate_external_reward(np.zeros(self.slot_space_size),
                                                                  self.current_slot_state, current_intent)
            else:
                reward = -self.w2 * self.calculate_external_reward(self.current_slot_state,
                                                                   np.ones(self.slot_space_size), goal=current_intent)
            # shifting the below part in the meta step acitons
            # self.current_intent_no += 1  # if all the intents in the current object are over

            goal_reached = True
            return self.current_slot_state, reward, goal_reached,

        # if not terminating action
        relevant_actions: List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot]) * 0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2 * self.calculate_external_reward(self.current_slot_state, new_state,
                                                              current_intent) - self.w1


        self.current_slot_state = np.copy(new_state)
        return self.current_slot_state, reward, False


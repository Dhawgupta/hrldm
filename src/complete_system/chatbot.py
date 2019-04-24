'''
author = Dhawal Gupta
Here I will implement the chatbot class
the chatbot class will be broken into 3 componenets as follows
1. NLU : This will contain both the intent model and teh slot model
2. DM : This will be the combination of a meta policy (action 5 not included as of now ) and a controller policy
3. NLG : This will take in input a action value from the DM (Controller actions) and output a sentence as well as the action

Controller action of -1 will denote a general intent, where the agent is interarcting in the meta domain
'''

import sys, os
import numpy as np
import random
from src.DQN.DQN1 import DQNAgent
from src.util import impdicts
from NLU_env import NLU_Simulator
from src.util import utils
from intent_module import intent_module
from arguments import get_args
from NLU_model import NLU
# from nlu_ic_model import NLU_IC
from src.util.utils import bcolors


printS = bcolors()

args = get_args()

NO_SLOTS = args['number_slots']
NO_INTENTS = args['number_intents']
META_OPTION_SIZE = args['number_options']
CONTROLLER_ACTION_SIZE = args['controller_action']
META_STATE_SIZE = NO_SLOTS + NO_INTENTS
CONTROLLER_STATE_SIZE = NO_SLOTS + NO_INTENTS


class Chatbots():
    def __init__(self, slot_space_size = 8, options_space = 5,  primitive_action_space = 20 , intent_space = 5, nlu_model  ='nlu.h5', intent_model = 'intent_module.h5', meta_agent_policy = "meta_agent.h5", controller_agent_policy = "controller_agent.h5"): # option_space is higly dependet, if we have option 5 then the space size will become 6
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.intent_space = intent_space
        self.slot_confidence = np.zeros(self.slot_space_size)
        self.intent_state  = np.zeros(self.intent_space)
        self.guessed_slots = {}
        self.previous_action = -1 # this contains the previous action taken
        # this will take care of teh current option being served
        # -1 tells that no current option is valid and the meta policy is tkaig actions
        self.current_option =-1 #
        self.previous_option = -1
        for tag in impdicts.all_tags:
            self.guessed_slots[tag] = "Unknown"

        # todo we need to put the intent model and the NLU model in one
        self.nlu = NLU()
        self.meta_agent = DQNAgent(state_size=META_STATE_SIZE, action_size=META_OPTION_SIZE, hiddenLayers=[75], dropout= args['dropout'], activation = 'relu', loadname = None, saveIn = False, learningRate= 0.05, discountFactor= 0.7, epsilon = 0.0)
        self.meta_agent.load(meta_agent_policy)
        self.controller_agent = DQNAgent(state_size=CONTROLLER_STATE_SIZE, action_size=CONTROLLER_ACTION_SIZE, hiddenLayers=nodes_hidden, dropout=0.000, activation='relu', loadname=None, saveIn=False, learningRate=0.05,discountFactor=0.7,epsilon=0.0)
        self.controller_agent.load(controller_agent_policy)
        self.nlg = NLG()
        self.intents_already_served = []
        print("Hello to the chatbot system how may I help you  ?")

    def step(self, sentence : str) : 
        """This will be responsibel for perfomring one step by taking in the sentence uttered by the user extrtacting relevant infromation and then responding back with an action and a sentence
        
        Arguments:

            sentence {str} -- The sentence uttered by the usr
        Return: action, sentence (nlg) and the guessed slots(dict) [reply, action, guessed slot values]
        # fixme when the controller action is -2 we will quit the interaction


        """
        if self.current_option == -1:
            if self.previous_action == -1:
                # check for no
                # fixme the final quitting of the whole agent
                if "no" in sentence.lower():
                    return ["Okay Bye", -2, self.guessed_slots]

                # we need to parse the sentence and the identify the intents as well as update the slots vlaeu
                self.intent_state = self.get_intent(sentence)
                # check if some of them are already served and then move onto next
        reply = "" # this is teh reply that step will produce
        if self.previous_action in range(11, 19):
            # it is a reasj action we have to check for a yes or no
            if "yes" in sentence.lower():
                self.slot_confidence[self.previous_action-11] = 1
            else:
                self.slot_confidence[self.previous_action-11] = 0
                self.guessed_slots[impdicts.position2tags[self.previous_action-11]] = "Unknown"

        elif self.previous_action in range(0,11):
            [labels , prob_values] = self.nlu.parse_sentence(sentence)
            tags = [impdicts.position2tags[p] for p in impdicts.action2slots[self.previous_action]]
            sentence_broken = sentence.split(' ')
            slot_values = {}
            slot_values_confidence = {}
            for t in tags:
                slot_values[t] = []
                slot_values_confidence[t] = []
            for t in tags:
                for i in range(len(labels)):
                    if labels[i] == t:
                        slot_values[t].append(sentence_broken[i])
                        slot_values_confidence[t].append(confidence[i])
            for t in tags:
                self.guessed_slots[t] = " ".join(slot_values[i])
                self.slot_confidence[impdicts.tags2position[t]] = sum(slot_values_confidence[t])/float(len(slot_values_confidence[t]))

        elif self.previous_action == 19:
            # todo have to decide how to act in this situation
            # nullify the currnt option state
            self.intent_state[self.current_option] = 0
            self.previous_option = self.current_option
            self.current_option = -1


        # now we move one step further by calling the move to the next state function
        # I think this needs to shifted to teh top
        # an options and action of -1 will denote an interactiong in teh meta domain about the user intents etc
        if self.current_option == -1:
            # then we use the meta policy

            # check if all the options are done ?
            if np.sum(self.intent_state) < 0.1:
                # ie no intents are active ask for the next set of intent
                reply = "Anthing else that I can help with ? "
                self.previous_action = -1
                return [reply, self.previous_action , self.guessed_slots]




            # else contionue with the next intent
            state = np.concatenate(self.slot_confidence , self.intent_state)
            state = state.reshape([1, META_STATE_SIZE])
            # meta_state = sta,te.copy()
            option = self.meta_agent.act(state, all_act = list(range(META_OPTION_SIZE)), epsilon = 0.0)
            # self.previous_option = self.current_option
            self.current_option = option


        goal_vector = utils.one_hot(self.current_option, NO_INTENTS)
        controller_state = np.concatenate([self.slot_confidence, goal_vector])
        controller_state = controller_state.reshape(1, CONTROLLER_STATE_SIZE)
        action = self.controller_agent(controller_state, all_act = list(range(CONTROLLER_ACTION_SIZE)), epsilon = 0)
        self.previous_action = action
        # get the reply from nlg
        reply = self.nlg.generate_sentence(controller_action= action, guessed_slot_values= self.guessed_slots)
        return [reply, self.previous_action, self.guessed_slots]


    # def get_slots_confidence(slots, confidence):
    #     """The given input has the original atis slots we need to replace them with our tags after that , we have to modigy the guessed slots values and the confidence state to refect the
    #
    #     Arguments:
    #         slots {[type]} -- [description]
    #         confidence {[type]} -- [description]
    #     """

    def get_intent(self, sentence):
        """THis will return the set of intents identified in a sentence

        Arguments:
            sentence {string} -- THe sentence spokin by user
        Return  :
            Returns the one hot vector telling us about the intents
        """
        if sentence is not None:
            sentence = sentence.lower()
            intent = intent_module(sentence)
            intents = impdicts.intents2index[intent]
            return utils.multi_hot(intents, NO_INTENTS)
        else:
            raise Exception("Some Error in dialogue reading")



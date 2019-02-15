'''
This code focuses on training the meta policy for the DM that we have proposed.
This implementation will focus on training the meta [polciy for a single controller network for multiple intent policies.
The intent weights files need to be of the format
weights_1.h5 for intent1 (or number 2 intent)
Points
1. This is based for singel neural network as the controller polciy
2, In this implementation we will discrad the previosu intent (many hot) vector and move onto the next vector


'''

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from src.DQN.DQN1 import DQNAgent
from collections import namedtuple
from src.util import impdicts
from src.envs.environments import MetaEnvMulti
from src.util import utils
from time import sleep
from datetime import datetime
from typing import List, Tuple, Dict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-cw','--controller-weights',required = True, help = "Weight for the controller policy model (i..e Mention  the file from whcih we need to load the qweights)")
# add the training parameters
ap.add_argument('-e','--episodes',required = False ,help='The number of episodes that the Meta Policy should train for.) If not specified then a default method will be used to save the file', type = int)
ap.add_argument('-mw','--meta-weights', required=False, help = "The Weights of the meta policy to be used for training (i.e. the file tobe saved for the same")

ap.add_argument('-gpu','--set-gpu', required=False,help ='Give the GPU number, if GPU is to be used, otherwise dont mention for run on CPU', type = int)

ap.add_argument('-lr','--learning-rate',required=False, default=0.05,type = float, help= 'The learning rate for meta policy training')
ap.add_argument('-df','--discount-factor',required = False, default = 0.7, type = float , help = 'THe Discount factor for the learning of meta polciy')
ap.add_argument('-do','--dropout',required = False,default = 0.00, type = float, help = 'The dropout probabliltiy for the meta policy training')

ap.add_argument('-ch', '--controller-hidden', required = False, help = "The Hidden Layers  configuration of the Controller Policym This portion is not yet implemeneted in the Code \nFormat : n1_n2_n3 ..  , tells n1 nodes in layer 1 , n2 nodes in layer 2 , n3 nodes in layer 3 and so on.", type = int , default = '75')
ap.add_argument('-ns','--number-slots',required=False , help = 'The number of confidence slots in the domain', type = int, default = 8)
ap.add_argument('-ni','--number-intents',required = False, help = 'The Number of intents that the domain has', type = int, default = 5)
ap.add_argument('-no','--number-options',required=False, help='The number of options for the meta policy', type = int, default = 6)
ap.add_argument('-ca','--controller-action', required = False, help = 'The Number of Controller Actions, although thte code is designed to work for only 20 action as of now', type = int, default = 20 )

args  = vars(ap.parse_args())

# TODO build an argparse for the weights folder

NO_SLOTS = args['number-slots']
NO_INTENTS = args['number-intents']
META_OPTION_SIZE = args['number-options']
CONTROLLER_ACTION_SIZE = args['controller-action']
META_STATE_SIZE = NO_SLOTS + NO_INTENTS
CONTROLLER_STATE_SIZE = NO_SLOTS + NO_INTENTS



def main():

    epsilon = 1
    env = MetaEnvMulti()  # TODO
    EPISODES = 300000 # To set this accordingly
    a = str(datetime.now()).split('.')[0]
    MetaAgent = DQNAgent(state_size=META_STATE_SIZE ,action_size= META_OPTION_SIZE, hiddenLayers=[75], dropout = args['dropout'], activation = 'relu',loadname = None, saveIn = False, learningRate=args['learning-rate'], discountFactor= args['discount-factor'])

    if 'meta-weights' not in args.keys():
        filename = "./save/Meta_    "
        filename = "_{}_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_all_intents_in_one.h5".format(filename, a ,str(MetaAgent.hiddenLayers), str(MetaAgent.dropout) , str(MetaAgent.learning_rate), str(MetaAgent.gamma), MetaAgent.activation, str(EPISODES))
    else:
        filename = args['meta-weights']

    # See if the user has given the hidden layer configuration for option_agnet
    nodes_hidden = [75] # defaulr value
    if 'controller-hidden' in args.keys():
        controller_hidden_config = args['controller-hidden']
        # extract the array from same
        nodes_hidden = [int(node) for node in controller_hidden_config.split('_') ]
    # load the agents for the controller , which is single for this case
    option_agent : DQNAgent = DQNAgent(state_size=CONTROLLER_STATE_SIZE ,action_size= CONTROLLER_ACTION_SIZE, hiddenLayers=nodes_hidden, dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.7 )# Not mkaing an agent for the user based actions
    option_agent.load(args['controller-weights']) # Load the weight for al the controller policies

    visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    batch_size = 64
    track = []
    i = 0
    for episode in range(EPISODES):
        # Now sample the next set of options from env
        goal = np.random.randint(META_OPTION_SIZE) # randomly sample a option to pursue
        running_reward = 0
        [confidence_state, intent_state] = env.reset() #
        # the intent state has multiple intents set as positive
        # visits[episode_thousand][state] += 1
        done = False # Running the meta polciy
        while not done:  # The Loop i whcih meta policy act
            all_options = env.constrain_options()
            state = np.concatenate([confidence_state, intent_state])

            state = state.reshape([1, META_STATE_SIZE])  # Converted to appropritate size
            meta_start_state = state.copy()

            option = MetaAgent.act(state, all_options, epsilon=epsilon) # TODO handle the 6 option chocie
            next_confidence_state = env.meta_step_start(option)  # get the reward at the sub policy level
            #############################################################
            # HERE COMES THE PART FOR CONTROLLER EXECUTION
            option_completed = False
            # make a one hot goal vector
            goal_vector = utils.one_hot(option, NO_INTENTS)
            while not option_completed:
                opt_actions = range(CONTROLLER_ACTION_SIZE) # Currently it is the while possible actions size
                controller_state = np.concatenate([next_confidence_state, goal_vector])
                controller_state = controller_state.reshape(1, CONTROLLER_STATE_SIZE)
                action = option_agents[option].act(controller_state, all_act = opt_actions, epsilon = 0 ) # provide episone for greedy approach
                next_confidence_state, _, option_completed = env.controller_step(option, action)
                next_controller_state = np.concatenate([next_confidence_state, goal_vector])
                next_controller_state = np.reshape([1, CONTROLLER_STATE_SIZE])
                # we dont need to store the experience replay memory for the controller policy
            next_confidence_state = next_confidence_state
            ###############################################

            confidence_state, next_confidence_state, intent_state, meta_reward , done = env.meta_step_end(option)
            meta_end_state = np.concatenate([next_confidence_state, intent_state])

            meta_end_state = meta_end_state.reshape([1, META_STATE_SIZE])
            epsilon = MetaAgent.observe((meta_start_state, option,meta_reward, meta_end_state ,done), epsilon= epsilon )
            if MetaAgent.memory.tree.total() > batch_size:
                MetaAgent.replay()
                MetaAgent.rem_rew(meta_reward)
            i += 1
            running_reward = running_reward + meta_reward
            if i % 100 == 0:  # calculating different variables to be outputted after every 100 time steps
                avr_rew = MetaAgent.avg_rew()
                track.append([str(i) + " " + str(avr_rew) + " " + str(episode) + " " + str(epsilon)])
                with open("results_" + a + "_.txt", 'w') as fi:
                    for j in range(0, len(track)):
                        line = track[j]
                        fi.write(str(line).strip("[]''") + "\n")
            # print(track)
            if done:
                print("episode: {}/{}, score: {}, e's: {}".format(episode, EPISODES, running_reward, epsilon))
                print("The state is : ", meta_end_state)
                break

            confidence_state = next_confidence_state

        if episode % 100 == 0:
            print("Episodes : {}".format(episode))
            # Saving the progress
            print("Saving")
            # convert this to save model for each policy
            agent.save(filename)
            # agent.saveController(fileController)
            sleep(0.2)
            print("Done Saving You can Now Quit")
            sleep(1)




if __name__ == "__main__":
    DQNAgent.setup_gpu('6')
    main()

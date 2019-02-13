'''
This code focuses on training the meta policy for the DM that we have proposed.
This implementation will focus on training the meta [polciy for a single controller network for multiple intent policies.
'''


import numpy as np
from ..DQN.DQN1 import DQNAgent
from collections import namedtuple
from ..util import impdicts
from ..envs.environments import MetaEnvMulti
from ..util import utils
from time import sleep
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.abspath('..'))

NO_SLOTS = 8
NO_INTENTS = 5
META_STATE_SIZE = NO_SLOTS + NO_INTENTS
META_OPTION_SIZE = 6
CONTROLLER_STATE_SIZE = NO_SLOTS + NO_INTENTS
CONTROLLER_ACTION_SIZE = 20


def main():
    filename = "./save/Meta_"
    epsilon = 1
    env = MetaEnvMulti()  # TODO
    EPISODES = 300000 # To set this accordingly
    a = str(datetime.now()).split('.')[0]
    MetaAgent = DQNAgent(state_size=META_STATE_SIZE ,action_size= META_OPTION_SIZE, hiddenLayers=[75], dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.7 )
    filename = "_{}_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_all_intents_in_one.h5".format(filename, a ,str(MetaAgent.hiddenLayers), str(MetaAgent.dropout) , str(MetaAgent.learning_rate), str(MetaAgent.gamma), MetaAgent.activation, str(EPISODES))


    visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    batch_size = 64
    track = []
    i = 0
    for episode in range(EPISODES):
        # Now sample the next set of options from env
        goal = np.random.randint(META_OPTION_SIZE) # randomly sample a option to pursue
        running_reward = 0
        [confidence_state, intent_state] = env.reset() # now the intent state can have multiple intents in there
        # visits[episode_thousand][state] += 1
        done = False
        while not done:  # The Loop i whcih meta policy act

            all_options = env.constrain_options()
            state = np.concatenate([confidence_state, intent_state])

            state = state.reshape([1, META_STATE_SIZE])  # Converted to appropritate size
            meta_start_state = state.copy()

            option = MetaAgent.act(state, all_options, epsilon=MetaAgent[goal])
            next_confidence_state = env.meta_step_start(option)  # get the reward at the sub policy level
            ##############################################
            # TODO Run the controller policy over here
            # while terminate_option:
            #   run ..
            #   run ..
            #
            #
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
            # print("Episod=" + str(episode))
            # print("i=" + str(i))

            # print("State : {}\nAction : {}\nNextState : {}\n".format(state,action,next_state))
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


META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS
CONTROLLER_ACTION_SIZE = 20
option = 1
import argparse




def run(option = option, episodes = 100000, no_slots = NO_SLOTS,meta_state_size = META_STATE_SIZE,meta_option_size = META_OPTION_SIZE,controller_state_size = CONTROLLER_STATE_SIZE,controller_action_size = CONTROLLER_ACTION_SIZE):
    print("Description of Run :\nOption : {}\nEpisodes : {}".format(option, episodes))
    sleep(1)
    filename = "./save/"
    epsilon = 1
    env = ControllerEnv()
    EPISODES = episodes
    a = str(datetime.now()).split('.')[0]

    agent = DQNAgent(state_size=controller_state_size ,action_size= controller_action_size, hiddenLayers=[75], dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.7 )


    filename = "{}{}_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_Childoption_{}.h5".format(filename, a ,str(agent.hiddenLayers), str(agent.dropout) , str(agent.learning_rate), str(agent.gamma), agent.activation, str(EPISODES), option)

    visits = 0

    batch_size = 64
    track = []
    i = 0
    for episode in range(EPISODES):
        # print("\n\n### EPISODE " + str(episode) + "###")
        goal = option # randomly sample a option to pursue
        visits +=1
        running_reward = 0
        [confidence_state, goal_vector] = env.reset(goal = goal)
        done = False
        while not done:  # The Loop i whcih meta policy acts
            all_acts = env.constrain_actions() # Still function will constrain the set of actions that are required

            state = np.array(confidence_state) # not using the goal vector
            state = state.reshape([1, controller_state_size])  # Converted to appropritate size
            action = agent.act(state, all_acts, epsilon=epsilon)
            next_confidence_state, reward, done = env.step(action)  # get the reward at the sub policy level
            next_state = np.array(next_confidence_state)
            next_state = next_state.reshape([1, controller_state_size])
            epsilon = agent.observe((state, action,reward, next_state,done), epsilon= epsilon)
            if agent.memory.tree.total() > batch_size:
                agent.replay()
            agent.rem_rew(reward)
            i += 1
            running_reward = running_reward + reward
            # print("Episode=" + str(episode))
            # print("i=" + str(i))

            if i % 1000 == 0:  # calculating different variables to be outputted after every 100 time steps
                print("Description of Run :\nOption : {}\nEpisodes : {}".format(option, episodes))
                print("i=" + str(i))
                avr_rew = agent.avg_rew()
                track.append([str(i) + " " + str(avr_rew) + " " + str(episode) + " " + str(epsilon)])
                with open("results_" + a + "_Policy_{}_.txt".format(goal), 'w') as fi:
                    for j in range(0, len(track)):
                        line = track[j]
                        fi.write(str(line).strip("[]''") + "\n")
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, running_reward, epsilon))
                print("The state is : ", next_state)
                break

            confidence_state = next_confidence_state
        if episode % 100 == 0:
            print("Episodes : {}".format(episode))
            # Saving the progress
            print("Saving")

            agent.save(filename)
            sleep(0.2)
            print("Done Saving You can Now Quit")
            sleep(1)





def main():
    """
    This will be the same function as above an will be used to call multiple iterations of programs with varying parameters

    :return:
    """
    DQNAgent.setup_gpu(6)

    parser = argparse.ArgumentParser(description='Run the trianing for a single intent')
    parser.add_argument('option', type=int, default=0,
                        help='an option to run')
    parser.add_argument('episodes', type=int, default=12000,
                        help='set the number of eipsodes to be run')

    args = parser.parse_args()
    # this will have parsed the arguments
    run(option= args.option, episodes=args.episodes)
    return 0

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    main()


def run_experiment():
    """
    This will be the same function as above an will be used to call multiple iterations of programs with varying parameters

    :return:
    """
    pass

if __name__ == "__main__":
    DQNAgent.setup_gpu('6')
    main()

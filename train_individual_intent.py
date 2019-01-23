import numpy as np
from collections import namedtuple
from DQN1 import DQNAgent
import impdicts
from environments import ControllerEnv
import utils
from time import sleep
from datetime import datetime


NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS + META_STATE_SIZE
CONTROLLER_ACTION_SIZE = 20


def main():
    filename = "./save/"
    # ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    # MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    epsilons = np.ones([META_OPTION_SIZE])
    env = ControllerEnv()  # TODO
    EPISODES = 12000
    agent = DQNAgent(state_size=CONTROLLER_STATE_SIZE ,action_size= CONTROLLER_ACTION_SIZE, hiddenLayers=[30,30,30], dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.9 )
    filename = "{}_{}_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}".format(filename, str(datetime.now()).split('.')[0],str(agent.hiddenLayers), str(agent.dropout) , str(agent.learning_rate), str(agent.gamma), agent.activation, str(EPISODES))
    # (filename + str(datetime.now()).split('.')[0] + str(agent.hiddenLayers) + u + str(agent.dropout)+ u +str(object=agent.learning_rate) + u + str(object=agent.gamma) + u + agent.activation + u + str(object=Episodes)+ ".h5"

    Visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    batch_size = 32
    track = []
    i = 0
    for episode in range(EPISODES):
        print("\n\n### EPISODE " + str(episode) + "###")
        # Get a random option
        goal = np.random.randint(META_OPTION_SIZE) # randomly sample a option to pursue
        running_reward = 0
        [confidence_state, goal_vector] = env.reset(goal = goal)
        # visits[episode_thousand][state] += 1
        done = False
        while not done:  # The Loop i whcih meta policy acts
            all_acts = env.constrain_actions() # Still function will constrain the set of actions that are required #TODO write this fcuntion constrain_actions(goal)
            # now we need to convert it to stat
            state = np.concatenate([confidence_state, goal_vector])
            state = state.reshape([1, CONTROLLER_STATE_SIZE])  # Converted to appropritate size
            action = agent.act(state, all_acts, epsilon=epsilons[goal])
            next_confidence_state, reward, done = env.step(action)  # get the reward at the sub policy level
            next_state = np.concatenate([next_confidence_state, goal_vector])
            next_state = next_state.reshape([1, CONTROLLER_STATE_SIZE])
            epsilons[goal] = agent.observe((state, action,reward, next_state,done), epsilon= epsilons[goal])
            if agent.memory.tree.total() > batch_size:
                agent.replay()
            agent.rem_rew(reward)
            i += 1
            running_reward = running_reward + reward
            print("e=" + str(e))
            print("i=" + str(i))

            # print("State : {}\nAction : {}\nNextState : {}\n".format(state,action,next_state))
            if i % 100 == 0:  # calculating different variables to be outputted after every 100 time steps
                avr_rew = agent.avg_rew()
                track.append([str(i) + " " + str(avr_rew) + " " + str(e) + " " + str(agent.epsilon)])
                with open("results1" + a + "_.txt", 'w') as fi:
                    for j in range(0, len(track)):
                        line = track[j]
                        fi.write(str(line).strip("[]''") + "\n")
            # print(track)
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, running_reward, agent.epsilon))
                print("The state is : ", next_state)
                break

            confidence_state = next_confidence_state




def run_experiment():
    """
    This will be the same function as above an will be used to call multiple iterations of programs with varying parameters

    :return:
    """
    pass

if __name__ == "__main__":
    main()

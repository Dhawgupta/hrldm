import numpy as np
from collections import namedtuple
from hDQN import hDQN
import impdicts
from environments import MetaEnv
import utils


NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS + META_STATE_SIZE
CONTROLLER_ACTION_SIZE = 20

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    env = MetaEnv() # TODO
    agent = hDQN()
    visits = np.zeros((12, 6)) # not required for me
    anneal_factor = (1.0-0.1)/12000
    print("Annealing factor: " + str(anneal_factor))
    anneal_start_meta = 6 # start the annealing after 6000 epsidoes 
    for episode_thousand in range(12):
        for episode in range(1000): # Loop for each epsidoe
            total_external_reward = 0
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            [confidence_state, intent_state] = env.reset() # TODO
            # visits[episode_thousand][state] += 1
            done = False
            goal_reached = False
            while not done and not goal_reached: # The Loop i whcih meta policy acts
                goal = agent.select_goal(intent_state)
                agent.goal_selected[goal] += 1
                print("Meta State: {} , Options Selected : {}".format(intent_state, goal))
                external_reward = 0
                goal_reached = False
                goal_start_state = confidence_state # the starting state in the beginning of a goal
                # Loop in which a sub policy acts for a goal
                while not done and not goal_reached: #TODO currently I am not using the goal_reached flag in my implmementation 
                
                    action = agent.select_move(confidence_state, utils.one_hot(goal, META_OPTION_SIZE), goal)
                    print("Goal : {}, State : {}, Action : {}".format(goal, confidence_state, action))
                    [next_confidence_state, intent_state] , intrinsic_reward, goal_reached , done = env.step(action) # get the reward at the sub policy level
                    # visits[episode_thousand][next_state-1] += 1
                    # intrinsic_reward = agent.criticize(goal, next_state)
                    # goal_reached = next_state == goal
                    # if goal_reached:
                    #     agent.goal_success[goal-1] += 1
                    #     print("Goal reached!! ")
                    # we will assume when the sdubgoal to be done we will calculate the extrinsic reward
                    exp = ActorExperience(confidence_state,  utils.one_hot(goal, META_OPTION_SIZE), action, intrinsic_reward, next_confidence_state, done)
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    confidence_state = next_confidence_state
                # calculate the Meta Reward
                goal_end_state = confidence_state # the ending state in a goal
                external_reward = env.calculate_external_reward(goal_start_state, goal_end_state, goal)
                exp = MetaExperience(goal_start_state,  utils.one_hot(goal, META_OPTION_SIZE), external_reward, goal_end_state, done)
                agent.store(exp, meta=True)
                total_external_reward += external_reward                
                #Annealing 
                if episode_thousand > anneal_start_meta:
                    agent.meta_epsilon -= anneal_factor
                # avg_success_rate = agent.goal_success[goal-1] / agent.goal_selected[goal-1]
                
                # if(avg_success_rate == 0 or avg_success_rate == 1):
                #     agent.actor_epsilon[goal-1] -= anneal_factor
                # else:
                #     agent.actor_epsilon[goal-1] = 1- avg_success_rate
                agent.actor_epsilon[goal] -= anneal_factor
                if(agent.actor_epsilon[goal] < 0.1):
                    agent.actor_epsilon[goal] = 0.1
                print("meta_epsilon: " + str(agent.meta_epsilon))
                print("actor_epsilon " + str(goal) + ": " + str(agent.actor_epsilon[goal-1]))
                
            if (episode % 100 == 99):
                print("Episodes : {}".format(episode + episode_thousand*1000))
                # print(str(visits/1000) + "")
if __name__ == "__main__":
    main()

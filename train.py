import numpy as np
from collections import namedtuple
from hDQN import hDQN
import impdicts
from environments import MetaEnv



def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    env = MetaEnv()
    agent = hDQN()
    visits = np.zeros((12, 6))
    anneal_factor = (1.0-0.1)/12000
    print("Annealing factor: " + str(anneal_factor))
    for episode_thousand in range(12):
        for episode in range(1000):
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            state = env.reset()
            visits[episode_thousand][state-1] += 1
            done = False
            while not done:
                goal = agent.select_goal(one_hot(state))
                agent.goal_selected[goal-1] += 1
                print("\nNew Goal: "  + str(goal) + "\nState-Actions: ")
                total_external_reward = 0
                goal_reached = False
                while not done and not goal_reached:
                    action = agent.select_move(one_hot(state), one_hot(goal), goal)
                    print(str((state,action)) + "; ")
                    next_state, external_reward, done = env.step(action)
                    visits[episode_thousand][next_state-1] += 1
                    intrinsic_reward = agent.criticize(goal, next_state)
                    goal_reached = next_state == goal
                    if goal_reached:
                        agent.goal_success[goal-1] += 1
                        print("Goal reached!! ")
                    if next_state == 6:
                        print("S6 reached!! ")
                    exp = ActorExperience(one_hot(state), one_hot(goal), action, intrinsic_reward, one_hot(next_state), done)
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_external_reward += external_reward
                    state = next_state
                exp = MetaExperience(one_hot(state), one_hot(goal), total_external_reward, one_hot(next_state), done)
                agent.store(exp, meta=True)
                
                #Annealing 
                agent.meta_epsilon -= anneal_factor
                avg_success_rate = agent.goal_success[goal-1] / agent.goal_selected[goal-1]
                
                if(avg_success_rate == 0 or avg_success_rate == 1):
                    agent.actor_epsilon[goal-1] -= anneal_factor
                else:
                    agent.actor_epsilon[goal-1] = 1- avg_success_rate
            
                if(agent.actor_epsilon[goal-1] < 0.1):
                    agent.actor_epsilon[goal-1] = 0.1
                print("meta_epsilon: " + str(agent.meta_epsilon))
                print("actor_epsilon " + str(goal) + ": " + str(agent.actor_epsilon[goal-1]))
                
            if (episode % 100 == 99):
                print("")
                print(str(visits/1000) + "")
if __name__ == "__main__":
    main()

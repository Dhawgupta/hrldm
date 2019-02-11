## This contains the basic structure for code
### Date : 14/1/19 
Working on hDQN with following specs :
**Meta Policy** :
- State : The all possible intents i.e. size of 5 for each intent
- Action : ALl the options (i.e. policy for each intent)
**Controller Policy** : 
- State : Intent space (5) + ALl slots (8) = 5 + 8 => 13
- Action : All the primitive action (8[ask each slot] + 3[hybrids] + 8[reask slots] + 1[terminate task] )
- Point : Currently I am not restructing the controller to take actions that might not be relevant to the controller (hence airfare controller will be able to take actions regardign ground service also , whoch can be made easier by restricing the controller )
- Point 2 : When the agent takes an action not meant for a specific goal the state does not change and it gets a negative iteration award (in the current model)

### Reward
**Extrinsic Reward for the Meta Policy**
1. Case 1: We can just calculate the total probability change from $s_t$ to $s_{t_k}$, Problem : Problem with this approach is that there is not penalty for long action, but at the same.
Currently there is not negative reward for picking up a wrong policy (We can change this by negating the number of steps in each loop).

**Intrisic Reward**
1. Currently I am planning to keep the intrinsic reward the same from the earlier cases.


### Date : 16/1/19
1. I have implemented the `train.py` which will train the neural net , it seems fine as of now but I havent run it yet.
2. I am yet to implement the enviornment of the agent.
Thoughts I am having
1. Suppose a person inquired about the ground service like transport type like the agent asks, and the human replies as follows
"What transport type do you require"
"What all transport is available"
At this point our agent should be able to show him the appropraite transport available which can only be done if the agent has the context of what it asked in the question , for that it needs to retain its last action.
So we can have a system , where the NLU is able to detect Answer / Question in the reply and approriately take the value or give the options.
This will need other set of actions which will include showing the availabel transport, (in here also there can be 2 cases) : 
Case 1 : The Agent knows the Ground Service City and can put the appropraite options forward
Case 2 : The agent does not know the transport city and will put the options forward after asking the appropriate city. (if this also becomces a request slot , then there can be an issue in this )

### Date : 20/1/19
1. I think we need to separate the `internal_step` and the `external_step` functions in the enviornmental , because what is happening that the supposed for a given intent `i` in meta policy, it selects and option `j`. Now when we enter the step function , I use the goal from the env pre set values. i.e. the goal is set to `i` and bacuse of this actions pertaining to policy of `j` when taken  are not propely rewarded. **1**. So what we need to do is separate the step functions of the meta-action and primitive actions.

2. **Line of action** : So when we call the meta.predict function, this will return an option to be followed. After selecting that option we will use the toption selected in all the internal_step transition as an input only. After the end of the sub goal we will take a step from `meta-step` function, and return the appropriate reward


### Date : 22/1/19
1. I am not having much idea on how to check the progress of the training procdure, I am planning to build and `ControllerEnv` for each sub goal so as to train each goal separately. This will take as input the goal to pursue and have the same primitive action set as the Meta Environment



### Date : 23/1/19
#### Programs being developed : 
 1. `environments.py`
 2. `train_individual_intent.py` : training procdure for each intent and basically training the controller policy.
 3. `DQN1.py` : using this code for training individual nets and inreality it is the DDQN-pER code  
#### Comments
1. Continuing with the yesterday approach, I have introduced another class in the `environments.py` class namely `ControllerEnv` which can basically simulate any subgoal for a training cycle. 
2. Meeting, Today where we discussed the possiblity on how to check the training progress of the model.
    2.1 I will complete the saving part of the Meta and Controller Policy
    2.2 Maam will figure out a way to check the trainig is happening properly or not
   
### Date : 24/1/19
#### Chnages to 
 1. `hDQN.py`
 2. `train.py`
 #### Comments
 1. I have implememnted a separaet saving and loading facility for both Meta and Controller Policy similar to our DQN approach and also changed the training code to integrate saving the policy every 100 episodes.
 2. Warning : We need to come with a proper naming style , for the time being I am using the Date : TIme .h5 as my format. 
 3. Also Currently I am saving the target policy of both and ignoring the normal actor.
<!-- # To Continue 
Continie from line 60 in the train co

de and set an appropritate annealing factor for the meta and controller policy which can take into account the intital bad controller policies and and hence have low annealing factor in the starting but as a the training progress the annesling factor adjusts accordingly -->

### Date : 28/1/19
Started thinking about the multi intent situation
Poins to be considered
1. The state space of the meta policy will now include the `intent space variable` as well the `current confidence values`
2. This will allow the meta policy to learn to pick the proper subpolci given the current confidence of the state, like 
if the slots respective to a intent are already filled the meta policy wont take the burden of callling that option just to quit immediately after that
3. WE need to add an additional option now , which says about starting to interact with the user to get the next set of intents, or to end the dialgoue

## Date : 29/1/19
Currently I will be implementing all the intents in a single controller policy later , we can separate the polcicies
for each intent into a separate neural net model.

## Date : 30/1/19
Codes working on 
- hDQN_multi.py
- train_multi.py
- environements.py : class `MetaEnvMulti` 

These codes will implements a single contreoller network for all the intents

Differences from the singel intent model.
1. Now the state space of meta controller contains the intents (which is not one hot anymore, but marks all intents that have to be servec)
along with the confidence values for all the slots
2. There will be an additional option to start a user interatciton to get the new set of intents.
3. (Debug) THe intent space will maintain all the history of intent, i.e. after picking the user_interaction option, when 
we get the new intents, we wont erase the older intents, rather than that we will add the new intents. It will be respoonsibiltiy of the meta contoller to decide to pick up the relevant intent.

Points not certain about
1. Currently I am making an option `user_agnet` whcich will be responsible for interacting with the user to get the next set of intents. THE PROBLEM : Should it be treated as one step option, i.e. a primitve action and directly feeded to the environement in the training process, or as multi step option, i.e the controller level poliy should take care of the asking the user part (this implementation will require extending the user action space by a lot).

Changes in environement needed:
1. First of we need to write a function that assignes intents, based on batches : IDEA : I will keep the random, intent generation as same and then randomly generate clubs of intents

Example

```buildoutcfg
intents = [4,3,1,0]
intent_grousps = create_intent_groups(intents)
>> intent_groups = [[4,3],[1,0]]

```
Over here the length of each sub group can be random

2. Also how do we club the intents, we can make default clubs whcih associate similar intents into a single group , but I my current implementation I will be making this clubbing also random

####Step to Continue: 
Left work at modification of the `MetaEnvMulti` class where I finished the implementaiton of the reset() function. I need to continue from chnaging the step function for meta and cotreoller to take into consideration the extra option added. I have replaced the `current_intent_no` , with the `current_intent_group_no` and need to repalce the same variable at the proper places appropriately Also check the implementation of the tarining code whcih seems to fine.


## 6/2/19
Started work on training individual models for each policy so that we can have train separaete policy for each intent. The file working on is `train_individual_intent_multi_model.py`

## 8/2/19
After the GPU maintainned I have put the code on run meanwhile I have been surveying some more of the work on end to end diaogue training and also studying a little bit of tensorflow to start understaningthe data effecient HRL codes'

## 9/2/18
Today I will start working on training all the models in a separate code , so that I dont have to rely on single code for the same.
I will be doing this under 
`train_individual_intent_multi_separate.py

## 11/2/19
Checked the training of single net for multiple intetnt under the 10/2 and seems to be converging as of now :D
Also I have organized the code better into folder. I need to add the NLU, NLG, and testing codes as of now to the main code.
Plans To be DOne : 
1. Put the Meta polciy on training on top of both the multiple net model and teh singel net model to see how it trains for serving multiple intents at a time. 
2. I also need to refactor the code now because of moving them into folder which we need to take care of now


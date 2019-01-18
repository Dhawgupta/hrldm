"""
This will contain helper functions
Description of the problem
Intents : 
0 : Flight 
1 : Airfare
2 : Airline
3 : Ground Service
4 : Ground Fare
Slots : 
0 : Departure City
1 : Arrival City
2 : Time
3 : Date
4 : Class
5 : Round Trip
6 : Ground City (We can replace this with the arrival city as well)
7 : Transport Type 
Actione : 
0 - 7 : Ask Each Slot (int the above order)
8 - 10 : Hybrid actions
    8 : Ask Dept City and Arr City
    9 : Time and Date
    10 : Ground City and Transport Type
11 - 18 : Reask Each slot value (to confirm if deemed requried)
19 : Terminate the conversation
"""

import numpy as np
import random
from impdicts import *

NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS + META_STATE_SIZE
CONTROLLER_ACTION_SIZE = 20




def get_random_action_goal(goal):
    """
    This code for a given goal will return a action (this action will not belong to another goal)
    Args :
    goal : Index specifying the intent
    """
    # check if goal is int , if not get the intent out
    if type(goal) == str: 
        goal = intent2indx[goal]
    return random.choice(intent2action[goal])
    


def one_hot(point, size): 
    '''
    Args: 
    point : The place where we want the one to be
    size : The size of the total vector
    returns : A one dimensional list (i.e. array) of the given number and ots vector
    '''
    vector = np.zeros(size)
    vector[point] = 1.0
    # return np.expand_dims(vector, axis=0) # returns a 2D array
    return vector
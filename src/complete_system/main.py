import random
from chatbot import Chatbots
from user import User
import argparse
from src.util import impdicts

ap = argparse.ArgumentParser()

ap.add_argument('--mode',default = "simulation", help = "Enter the mode of opearation for the chatbot")




def interact_human():
    '''
    This will be used to interact with a human
    :return:
    '''
    cbot = Chatbots()
    action = 0
    while action != -2:
        reply = input(">")
        [reply, action, slots] = cbot.step(reply)
        print(f"Action {action} : {reply} ")

    # done conversation

def interact_simulation():
    '''
    This will playout a simulationf for the uset
    :return:
    '''
    cbot = Chatbots()
    action = -3
    while action != -2:
        if action == -3 : # this means the greeing action




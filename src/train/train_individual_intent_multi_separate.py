'''
This code will be uised to run multiple intent training procesdfures
'''

import os
import subprocess

command = ' python train_general_intent.py 3 1'
subprocess.Popen(command)
for i in range(1000):
    print(i)
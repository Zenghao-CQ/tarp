import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from client import Environment,LoadPattern
import sys

sys.path.append("..")

from client import Environment
from ddpg import DDPG

# change this to the location of the checkpoint file
# CHECKPOINT_FILE = './checkpoints/manipulator/ep50.pth.tar'
CHECKPOINT_FILE = '../rl-tarp2/checkpoints/manipulator/ep100.pth.tar'

if __name__=="__main__":
    # environment for getting states and peforming actions
    lg=LoadPattern("guest",{"path":"../load/load3.csv"})
    # lg=LoadPattern("wave",{"max":80,"min":20,"period":40,"prop":0.07})
    env = Environment({},loadPattern=lg)

    # init ddpg agent
    agent = DDPG(env)
    
    # init from saved checkpoints
    agent.loadCheckpoint(CHECKPOINT_FILE)
    # start training
    agent.train()

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import os
import dgl

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic


plt.rc('font', family="Times New Roman")

mslist=['ts-basic-service', 'ts-seat-service', 'ts-station-service', 'ts-train-service', 'ts-travel-service']

PLOT_FIG = True

# Hyperparameters
ACTOR_LR = 0.00003
CRITIC_LR = 0.00003
MINIBATCH_SIZE = 64
NUM_EPISODES = 9000
NUM_TIMESTEPS = 500
MU = 0
# SIGMA = 0.2
SIGMA = 0.1
CHECKPOINT_DIR = './checkpoints/manipulator/'
BUFFER_SIZE = 100000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
EPSILON = 1.0
EPSILON_DECAY = 1e-6

NUM_ACTIONS = 12
NUM_STATES = 5
NUM_RES = 2

ID = 'default'

# converts observation dictionary to state tensor
# TODO: currently it's conversion between list and state tensor
def obs2state(state_list):
    #l1 = [val.tolist() for val in list(observation.values())]
    #l2 = []
    #for sublist in l1:
    #    try:
    #        l2.extend(sublist)
    #    except:
    #        l2.append(sublist)
    return torch.FloatTensor(state_list).view(1, -1)

class DDPG:
    def __init__(self, env):
        self.env = env
        self.stateDim = NUM_STATES
        self.actionDim = NUM_ACTIONS
        self.resDim = NUM_RES
        self.actor = Actor(self.stateDim, self.actionDim)
        self.critic = Critic(self.stateDim, self.actionDim)
        self.targetActor = deepcopy(Actor(self.stateDim, self.actionDim))
        self.targetCritic = deepcopy(Critic(self.stateDim, self.actionDim))
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []
        self.resourcegraph = []
        self.slograph = []
        self.alossgraph = []
        self.clossgraph = []
        self.delay2=[]
        self.delay1=[]
        self.start = 0
        self.end = NUM_EPISODES

    # Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
    # Target Q-value <- reward and bootstraped Q-value of next state via the target actor and target critic
    # Output: Batch of Q-value targets
    def getQTarget(self, nextStateBatch, rewardBatch):       
        targetBatch = torch.FloatTensor(rewardBatch)

        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        tmp = self.discount * qNext
        targetBatch = targetBatch + tmp
        return targetBatch
        # return Variable(targetBatch)

    # weighted average update of the target network and original network
    # Inputs: target actor(critic) and original actor(critic)
    def updateTargets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    # Inputs: Current state of the episode
    # Output: the action which maximizes the Q-value of the current state-action pair
    def getMaxAction(self, curState):
        action = self.actor(curState)
        max_idx_list = []
        for i in range(len(action)):
            if self.noise==None:
                actionNoise=action[i]
            else:
                noise = self.epsilon * Variable(torch.FloatTensor(self.noise()))
                actionNoise = action[i] + noise
                action[i]=actionNoise
            # get the max
            actionNoise=action[i]
            action_list = actionNoise.tolist()
            max_action = max(action_list)
            max_index = action_list.index(max_action)
            max_idx_list.append(max_index)

        return max_idx_list, action

    # training of the original and target actor-critic networks
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')
        
        action_step = 10
        available_actions = [0, action_step, -action_step]
        all_rewards = []
        avg_rewards = []
        # for each episode 
        for episode in range(self.start, self.end):
            # reset noise
            # if episode==201:
            #     self.noise=OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA*0.8)
            # if episode==301:
            #     self.noise=OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA*0.5)
            # if episode==401:
            #     self.noise=OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA*0.3)
            # if episode==501:
            #     self.noise=None
            #     self.replayBuffer=Buffer(BUFFER_SIZE)
            state = self.env.new_reset()
            self.delay1=[]
            self.delay1=[]
            self.slograph=[]
            self.resourcegraph=[]
            ep_reward = 0
            
            for step in range(NUM_TIMESTEPS):
                g,d,SLO_score,res_use = self.env.get_state() #state_graph, reawrd, delay, SLO, resource usage
                fin = g.ndata["N_DELAY"]
                fint = g.ndata["N"]
                fin=torch.cat([fin,fint],dim=1)
                print(fin.shape)
                # print each time step only at the last EPISODE
                if episode == NUM_EPISODES-1:
                    print("EP:", episode, " | Step:", step)
                    print("Update - Current SLO Retainment:", SLO_score)
                    print("Update - Current Util:", res_use)

                # get maximizing action
                self.actor.eval()     
                action, actionToBuffer = self.getMaxAction(fin)

                prof={}

                #make prof dict
                for i,k in enumerate(mslist):
                    prof[k]=(action[i]+1)*0.5

                if episode == NUM_EPISODES-1:
                    print("Update - Actions to take:", prof)

                self.actor.train()
                
                # step episode
                nextg,reward,d,SLO_score,res_use=self.env.perform_action(prof) #state_graph, reawrd, delay, SLO, resource usage
                nextfin = g.ndata["N_DELAY"]
                nextfint = g.ndata["N"]
                nextfin=torch.cat([nextfin,nextfint],dim=1)
                ep_reward = ep_reward + reward
                print('Reward: {}'.format(reward),'EP-Reward: {}'.format(ep_reward))
                
                #record resource
                self.resourcegraph.append(res_use.item())
                self.slograph.append(SLO_score)
                self.delay2.append(d[1])
                self.delay1.append(d[0])
                # Update replay bufer
                self.replayBuffer.append((fin, actionToBuffer, nextfin, reward))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)

                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch)
                    
                    with torch.autograd.set_detect_anomaly(True):
                        # Critic update
                        self.criticOptim.zero_grad()
                        criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                        loss1 = criticLoss.detach_().requires_grad_(True)
                        loss1.backward(retain_graph=True)
                        print('Critic Loss: {}'.format(criticLoss))
                        self.criticOptim.step()
                        
                        # Actor update
                        self.actorOptim.zero_grad()
                        actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                        actorLoss.backward(retain_graph=True)
                        print('Actor Loss: {}'.format(actorLoss))
                        self.actorOptim.step()
                        
                        if step % 60 == 0:
                            self.alossgraph.append(actorLoss.item())
                            self.clossgraph.append(criticLoss.item())
                    # Update Targets                        
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay
            print("EP -", episode, "| Total Reward -", ep_reward)
   
            # save to checkpoints
            if episode % 50 == 0:
                self.save_checkpoint(episode)
            self.rewardgraph.append(ep_reward.item())
            # print(self.rewardgraph)
            # print(self.resourcegraph)
            if PLOT_FIG:
                delta=20
                if episode<400:
                    delta = 10
                if  episode % delta ==0 and episode != 0:
                    plt.cla()
                    plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Episodes')
                    plt.savefig('./fig/ep'+str(episode)+'-reward.png')

                    plt.cla()
                    plt.plot(self.resourcegraph, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-res.png')

                    plt.cla()
                    plt.plot(self.slograph, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-slo.png')

                    plt.cla()
                    plt.plot(self.alossgraph, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-actor-loss.png')

                    plt.cla()
                    plt.plot(self.clossgraph, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-critic-loss.png')

                    plt.cla()
                    plt.plot(self.delay1, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-avgdelay.png')

                    plt.cla()
                    plt.plot(self.delay2, color='darkorange')  # total resource usage in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Steps')
                    plt.savefig('./fig/ep'+str(episode)+'-p95delay.png')
                    
                    f1=open("./logs/rewards-"+str(episode)+".csv",mode='w')
                    f1.write(",".join([str(x) for x in self.rewardgraph]))
                    f1.close()
                    f2=open("./logs/resource-"+str(episode)+".csv",mode='w')
                    f2.write(",".join([str(x) for x in self.resourcegraph]))
                    f2.close()
                    f2=open("./logs/slo-"+str(episode)+".csv",mode='w')
                    f2.write(",".join([str(x) for x in self.slograph]))
                    f2.close()
                    f2=open("./logs/avg-"+str(episode)+".csv",mode='w')
                    f2.write(",".join([str(x) for x in self.delay1]))
                    f2.close()
                    f2=open("./logs/p95-"+str(episode)+".csv",mode='w')
                    f2.write(",".join([str(x) for x in self.delay2]))
                    f2.close()
        if PLOT_FIG:
            plt.cla()
            plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.savefig('./fig/final.png')

            plt.cla()
            plt.plot(self.resourcegraph, color='darkorange')  # total resource usage in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Steps')
            plt.savefig('./fig/finel-res.png')

            plt.cla()
            plt.plot(self.slograph, color='darkorange')  # total resource usage in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Steps')
            plt.savefig('./fig/final-slo.png')

    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer,
            'rewardgraph': self.rewardgraph,
            'resourcegraph':self.resourcegraph,
            "clossgraph": self.clossgraph,
            "alossgraph": self.alossgraph,
            'epsilon': self.epsilon
            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.resourcegraph = checkpoint['resourcegraph']
            self.clossgraph = checkpoint['clossgraph']
            self.alossgraph = checkpoint['alossgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')

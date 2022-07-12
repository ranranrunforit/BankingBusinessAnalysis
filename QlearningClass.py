# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:53:58 2022

@author: Chaoran
"""

import numpy as np
import pandas as pd
import time
from queue import PriorityQueue
import datetime
import os
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#pd.set_option('mode.chained_assignment',None)

class DualPriorityQueue(PriorityQueue):
    def __init__(self,maxPQ=False):
        PriorityQueue.__init__(self)
        self.reverse = -1 if maxPQ else 1
    
    def put(self, priority, data):
        PriorityQueue.put(self,(self.reverse * priority,data))
        
    def get(self, *args,**kwargs):
        priority, data = PriorityQueue.get(self, *args,**kwargs)
        return self.reverse*priority,data
    
orderbook_data = pd.read_csv(r'Users//Downloads/LOBSTER_SampleFile_AAPL_2012-06-21_30/AAPL_2012-06-21_30.csv')
orderbook_data.index = [str(i) for i in range(len(orderbook_data))]
ori_data = orderbook_data.iloc[:]

MAX_EPISODES = 120
LOOP_PER_EPISODES = 100
ACTIONS = [str(x*100) for x in range(-10,30)] # actions
EPSILON = 0.9 # greedy
ALPHA = 1 # Study speed
GAMME = 9 # Decay
N_TIME_STATES = 6 # 6 decision making 6*5 = 30 sec for 1 trade 
N_INVENTORY_STATES = 4# 4 Unit to Execute  2500/4 shares/unit, trade either 625,1250,1875,2500 shares


class RL(object):
    def __init__(self,ACTION,EPSILON,ALPHA,GAMMA):
        
        self.ACTIONS = ACTIONS # actions
        self.EPSILON = EPSILON # greedy
        self.ALPHA = ALPHA # Study speed
        self.GAMMA = GAMMA # Decay
        self.q_table = pd.DataFrame(columns=self.ACTIONS,dtype=np.float64)
        
        #FRESH_TIME = 0.1
        
    def choose_action(self,state):
        
        state=str(state)
        self.check_state_exist(state)
        state_actions = self.q_table.loc[state, :]
        if (np.random.uniform()>self.EPSILON) or (state_actions.all() == 0):
            action_name = np.random.choice(self.ACTIONS)
        else:
            action_name = state_actions.argmax()
        return int(action_name)
    
    def check_state_exist(self,state):
        
        state=str(state)
        if state not in self.q_table.index:
            #append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.ACTIONS),
                        index = self.q_table.columns,
                        name=str(state)
                )        
            )
        print(self.q_table)
    
    def learn(self,*args):
        pass
    
class QLearningTable(RL):
    def __init__(self,ACTION,EPSILON,ALPHA,GAMMA):
        super(QLearningTable,self).__init__(ACTIONS,EPSTLON,ALPHA,GAMMA)
    
    def learning(self):
        # input A_ but it doesn't use it in Q learning
        self.check_state_exist(S_[:2])
        q_perdict = self.q_table.loc[str(S[:2]),str(A)]
        if S_[1]!=0:
            q_target = R + self.GAMMA * self.q_table.loc[str(S_[0:2]), :].max() # Q estimate
        else:
            q_target = R # next state is terminal
        
        self.q_table.loc[str(S[:2]),str(A)] += self.ALPHA * (q_target - q_predict) # add adjustment
        

def update_Qlearning():
    
    for k in range(LOOP_PER_EPISODES):
        for episode in range(MAX_EPISODES-1):
        # inital obervation
            data_eps_temp=ori_data[ori_data['Time']>(ori_data.ix[0,'Time']+episode*N_TIME_STATES*TIME_INTERVAL)]
            data_eps = data_eps_temp[ori_data['Time']>(ori_data.ix[0,'Time']+(1+episode)*N_TIME_STATES*TIME_INTERVAL)]
            S = [N_TIME_STATES,N_INVENTORY_STATES,INVENTORY]
            is_terminated = False
            
            while not is_terminated:
                A = RL.choose_action(str(S[0:2]))
                
                S_, R = get_env_feedback(5, A , data_eps)
                #Q learning
                RL.leang(S,A,R,S_,None)
                
                S = S_
                
                if S_[1]==0:
                    is_terminated=True
        RL.q_table.to_csv(r'User//Downloas/LOBSTER_SampleFile_AAPL_2012-06-21_30/Result.csv')
        print("One Loop")

DIRECTION='Buy'
LEVEL_ORDERBOOK=30
TIME_INTERVAL = 5 #every 5 seconds for 1 order
DDL=TIME_INTERVAL*N_TIME_STATES # 30 seconds to react
INVENTORY=2500 # total number of shares
UNIT=INVENTORY/N_INVENTORY_STATES # shares per unit

if __name__ == "__main__":
    RL = QLearningTable(ACTIONS,EPSILON,ALPHA,GAMMA)
    update_Qlearning()
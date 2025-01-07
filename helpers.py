# -*- coding: utf-8 -*-
"""
Created on May 10th, 2024

@author: Zhewei Zhang

zhzhewei36@gmail.com
"""

import pickle 
import numpy as np

# In[] bhv data

# Load behavioral data from GONOGO task
bhv_file = './training_set/GONOGO_Bhv.pickle'
with open(bhv_file, 'rb') as f:
    bhv_data = pickle.load(f)

# Initialize a dictionary to store variables
ratBhv_GONOGO = {'odor': [], 'action': [], 'reward': []}

# Process each rat's data
for rat, data in bhv_data.items():
    # Convert lists to numpy arrays for easier manipulation
    odor = np.array(data['odor'])
    action = np.array(data['action'])
    reward = np.array(data['reward'])
    
    # If an odor value is above 100, subtract 100 
    odor[odor > 100] = odor[odor > 100] - 100

    # Identify valid trials:
    idx = np.logical_not(np.isnan(odor)) & np.array([True if a in [4, 7] else False for a in action])
    
    # Filter the odor, action, and reward arrays based on these valid trials
    odor, action, reward = odor[idx], action[idx], reward[idx]

    # Prepare a list of reward outcomes
    reward_list = np.zeros(reward.size, dtype=object)
    reward_list[reward == 1] = 'reward'
    reward_list[(reward == 0) & (action == 4)] = 'absent'
    reward_list[(reward == 0) & (action == 7)] = 'timeOut'

    # Append the processed data to the ratBhv_GONOGO dictionary
    ratBhv_GONOGO['odor'].append(odor)
    ratBhv_GONOGO['action'].append(action)
    ratBhv_GONOGO['reward'].append(reward_list.tolist())



# Load behavioral data from reversal learning task
bhv_file_reversal = './training_set/reversal_bhv.pickle'
with open(bhv_file_reversal, 'rb') as f:
    bhv_data = pickle.load(f)

# Initialize a dictionary to store variables
ratBhv_reversal = {
    'odor': [],
    'action': [],
    'reward': [],
    'reversal_point': []
}

# Process each rat's data
for rat, data in bhv_data.items():
    # Convert lists to numpy arrays for easier manipulation
    odor = np.array(data['odor'])
    action = np.array(data['action'])
    reward = np.array(data['reward'])
    
    # If an odor value is above 100, subtract 100 
    odor[odor > 100] = odor[odor > 100] - 100
    
    # Identify valid trials:
    idx = np.logical_not(np.isnan(odor)) & np.array([True if a in [4, 7] else False for a in action])
    
    # Filter the odor, action, and reward arrays based on these valid trials
    odor, action, reward = odor[idx], action[idx], reward[idx]

    # Prepare a list of reward outcomes
    reward_list = np.zeros(reward.size, dtype=object)
    reward_list[reward == 1] = 'reward'
    reward_list[(reward == 0) & (action == 4)] = 'absent'
    reward_list[(reward == 0) & (action == 7)] = 'timeOut'
    
    # Append the processed data to the ratBhv_reversal 
    ratBhv_reversal['odor'].append(odor)
    ratBhv_reversal['action'].append(action)
    ratBhv_reversal['reward'].append(reward_list.tolist())
    
    # Identify the "reversal point" as the first occurrence of a 'reward'
    # where the odor matches the maximum odor value
    ratBhv_reversal['reversal_point'].append(
        np.where((reward == 1) & (odor == np.max(odor)))[0][0]
    )

# In[] Neuron / Fiber Photometry Data

# Set the path to the directory containing the pickled datasets
path = './training_set/'

# GONOGO task: Real ACh data
file = path + 'GONOGO_ACh.pickle'
with open(file, 'rb') as f:
    dataACh_GONOGO = pickle.load(f)

# Reversal task: Real ACh data
file = path + 'reversal_ACh.pickle'
with open(file, 'rb') as f:
    dataACh_reversal = pickle.load(f)


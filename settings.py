# -*- coding: utf-8 -*-
"""
Created on May 10th, 2024

@author: Zhewei Zhang

zhzhewei36@gmail.com
"""

import random
import numpy as np
from helpers import ratBhv_GONOGO, ratBhv_reversal


# In[]

# Function to retrieve a configuration dictionary for a given model_name
def get_configuration(model_name):
    
    # Define US (unconditioned stimulus) intensities for different outcomes
    lambda_US_reward, lambda_US_absent, lambda_timeOut = 1, -0.1, -0.2
    
    # Dictionary containing different model configurations
    configurations = {
        # Mackintosh model: alpha updates, sigma fixed
        "Mackintosh": {
            "range_alpha_update": True,
            "range_alpha": {k: [0.01, lambda_US_reward] for k in ['other_stimuli', 'odorA', 'odorB']},
            "range_sigma": [0.01, 0.01]
        },
        # Pearce-Hall model: sigma updates, alpha fixed range
        "Pearce-Hall": {
            "range_alpha_update": False,
            "range_sigma": [0.01, 1],
            "range_alpha": {k: [0.01, 0.1] for k in ['other_stimuli', 'odorA', 'odorB']}
        },
        # Hybrid model: both alpha and sigma update
        "Hybrid": {
            "range_alpha_update": True,
            "range_sigma": [0.01, 1],
            "range_alpha": {k: [0.01, lambda_US_reward] for k in ['other_stimuli', 'odorA', 'odorB']}
        }
    }

    # Modify upper limit of alpha for odorB in Mackintosh and Hybrid models
    # to reflect the absolute value of lambda_US_absent
    configurations["Mackintosh"]["range_alpha"]['odorB'][1] = abs(lambda_US_absent)
    configurations["Hybrid"]["range_alpha"]['odorB'][1] = abs(lambda_US_absent)

    # Extract the desired model configuration
    configuration = configurations[model_name]
    
    # Set proportion of alpha vs. sigma in overall salience calculation
    configuration["nfactor_alpha"] = 0.3
    configuration["nfactor_sigma"] = 1 - configuration["nfactor_alpha"]

    # Define additional parameters
    configuration["lr_v"] = 0.2  # Learning rate for updating V_A or V_A_inhib
    configuration["lr_alpha"] = {'up': 0.1, 'down': 0.1}
    configuration["lr_sigma"] = {'up': 0.1, 'down': 0.1}
    
    # Exponential decay factor for alpha's upper limit
    configuration["alpha_decey"] = 0.9995
    # US intensities in the configuration
    configuration["lambda"] = {
        'reward': lambda_US_reward,
        'absent': lambda_US_absent,
        'timeOut': lambda_timeOut
    }
    
    # Initial values for alpha, sigma, salience, V, and V_inhib
    configuration["init_alpha"] = 0.05
    configuration["init_sigma"] = 0.05
    configuration["init_salience"] = 0.05  # This might be updated before first trial
    # Initialize V as an average of reward and timeOut intensities
    configuration["init_V"] = (lambda_US_reward + lambda_timeOut) / 2 / 2
    configuration["init_V_inhib"] = 0

    # Noise parameters for V, alpha, and sigma
    configuration["noise_variance_v"] = 0.025
    configuration["noise_variance_alpha"] = 0.05
    configuration["noise_variance_sigma"] = 0.05
    
    # Baseline value for salience
    configuration["salience_baseline"] = 0.0
    
    return configuration


# Function to generate trials for either a GONOGO or reversal task
def get_trials(num, task='GONOGO', realBhvData=True):
    """
    Generate trial sequences for behavioral tasks (GONOGO or reversal).

    """
    # Initialize a list to store trial dictionaries
    trials = []

    # Handle GONOGO task generation
    if task.upper() == 'GONOGO':
        if realBhvData:
            # Use real data from ratBhv_GONOGO
            nth = random.choice(range(len(ratBhv_GONOGO['odor'])))
            odor = ratBhv_GONOGO['odor'][nth]
            reward = ratBhv_GONOGO['reward'][nth]
            
            # Create a sequence of num trials
            for i in range(num):
                # Avoid out-of-bound errors by re-sampling if i exceeds the length of odor array
                if i < len(odor):
                    i_ = i
                else:
                    i_ = random.choice(range(len(odor) - 40, len(odor)))
                
                # Assign 'odorA' or 'odorB' based on whether the odor at index i_ is the minimum value
                o = 'odorA' if odor[i_] == min(odor) else 'odorB'
                r = reward[i_]
                # Each trial includes 'other_stimuli' plus the selected odor, and the corresponding outcome
                trials.append({'CS': ['other_stimuli', o], 'US': r})
        else:
            # Generate synthetic data for GONOGO
            for i in range(num):
                if np.random.rand() > 0.5:
                    trials.append({'CS': ['other_stimuli', 'odorA'], 'US': 'reward'})
                else:
                    # 'timeOut' used for the alternate outcome
                    outcome = 'timeOut'
                    trials.append({'CS': ['other_stimuli', 'odorB'], 'US': outcome})
    
    # Handle reversal task generation
    elif task.lower() == 'reversal':
        if realBhvData:
            # Use real data from some reversal dataset (ratBhv_reversal)
            nth = random.choice(range(len(ratBhv_reversal['odor'])))
            odor = ratBhv_reversal['odor'][nth]
            reward = ratBhv_reversal['reward'][nth]
            reversal_point = ratBhv_reversal['reversal_point'][nth]

            # First set of trials before reaching the reversal point
            for i in range(num):
                if i <= reversal_point:
                    i_ = i
                else:
                    # After the reversal point, randomly sample near the reversal point
                    i_ = random.choice(range(reversal_point - 40, reversal_point))
                
                o = 'odorA' if odor[i_] == min(odor) else 'odorB'
                r = reward[i_]
                trials.append({'CS': ['other_stimuli', o], 'US': r})
            
            # Additional trials post-reversal (2 times more, as an example)
            for i in range(num * 2):
                if i + reversal_point < len(odor):
                    i_ = i + reversal_point
                else:
                    i_ = random.choice(range(len(odor) - 40, len(odor)))
                
                o = 'odorA' if odor[i_] == min(odor) else 'odorB'
                r = reward[i_]
                trials.append({'CS': ['other_stimuli', o], 'US': r})
        else:
            # Generate synthetic data for reversal
            # First phase
            for i in range(num):
                if np.random.rand() > 0.5:
                    trials.append({'CS': ['other_stimuli', 'odorA'], 'US': 'reward'})
                else:
                    outcome = 'timeOut'
                    trials.append({'CS': ['other_stimuli', 'odorB'], 'US': outcome})

            # Reversal phase (twice as many trials)
            for i in range(num * 2):
                if np.random.rand() > 0.5:
                    trials.append({'CS': ['other_stimuli', 'odorA'], 'US': 'absent'})
                else:
                    trials.append({'CS': ['other_stimuli', 'odorB'], 'US': 'reward'})

    return trials

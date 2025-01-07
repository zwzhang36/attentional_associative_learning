# -*- coding: utf-8 -*-
"""
Created on May 10th, 2024

@author: Zhewei Zhang

zhzhewei36@gmail.com
"""
import pickle
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from settings import get_configuration, get_trials
from helpers import dataACh_reversal, dataACh_GONOGO

# In[]

# Function to update V_A / V_A_inhib
def update_association(V_A, V_A_inhib, V_A_net, lr_v, 
                       lambda_, salience, RPE):
    """
    Update the excitatory (V_A) or inhibitory (V_A_inhib) associative strength 
    based on the reward prediction error (RPE) and salience.
    """
    
    # Update learning rate based on salience 
    lr_v_ = lr_v * salience

    # Update the excitatory or inhibitory component based on the sign of RPE
    if RPE > 0:
        V_A = V_A + lr_v_ * RPE
    else:
        V_A_inhib = V_A_inhib - lr_v_ * RPE
        
    # Enforce lower bound of 0 for both V_A and V_A_inhib
    V_A = 0 if V_A < 0 else V_A
    V_A_inhib = 0 if V_A_inhib < 0 else V_A_inhib
    
    return V_A, V_A_inhib

# Function to calculate net associative strength
def net_associative_strength(V_A, V_A_inhib):
    """
    Compute the net associative strength as the difference between V_A and V_A_inhib.
    """
    return V_A - V_A_inhib

# Function to update alpha
def update_alpha(alpha, V_A_net, V_X_net, lr_alpha, lambda_, range_alpha):
    """
    Update the alpha parameter based on the difference between lambda_ 
    and net associative strengths (V_X_net, V_A_net).
    """
    # Calculate the delta alpha
    delta_alpha = (abs(lambda_ - V_X_net) - abs(lambda_ - V_A_net)) 

    # Determine learning rate for alpha  
    lr_alpha_ = lr_alpha['up'] if delta_alpha>0 else  lr_alpha['down']

    # Update alpha
    alpha_new = alpha + lr_alpha_ * delta_alpha
            
    # Check the range limits for alpha
    if range_alpha[1] < range_alpha[0]:
        print('error')  # Inconsistent range
    # Enforce alpha range limits
    if alpha_new < range_alpha[0]:
        alpha_new = range_alpha[0]
    if alpha_new > range_alpha[1]:
        alpha_new = range_alpha[1]

    return alpha_new

# Function to update sigma
def update_sigma(sigma, V_net, lr_sigma, lambda_, range_sigma):
    """
    Update the sigma parameter based on the difference between lambda_
    and the net associative strength.
    """
    # Determine learning rate for sigma  
    lr_sigma_ = lr_sigma['up'] if abs(lambda_ - V_net) > sigma else lr_sigma['down']
        
    # Update sigma
    sigma_new = (1 - lr_sigma_) * sigma + lr_sigma_ * abs(lambda_ - V_net)
    
    # Enforce sigma range limits
    if sigma_new < range_sigma[0]:
        return range_sigma[0]
    elif sigma_new > range_sigma[1]:
        return range_sigma[1]
    else:
        return sigma_new

def update_threshold(alphas, alpha_decey, lambda_):
    """
    Update the threshold (upper limit) of alpha based on lambda_.
    This function ensures that if abs(lambda_) exceeds the current alpha_upper_limit, 
    it becomes the new limit. Otherwise, the limit is reduced by a decay factor.
    """
    alpha_lower_limit, alpha_upper_limit = alphas[0], alphas[1]
    alpha_upper_limit = max(alpha_upper_limit, alpha_lower_limit)
    
    if abs(lambda_) >= alpha_upper_limit:
        # If the magnitude of lambda_ is above the current alpha_upper_limit,
        # use that as the new limit
        return abs(lambda_)
    else:
        # Otherwise, decay the alpha_upper_limit
        return alpha_upper_limit * alpha_decey
    
# Simulation function for a learning model
def simulate_learning(trials, config, setting):
    # These variables store parameters that get updated across trials
    alphas, sigmas, saliences, V_As, V_A_inhibitories = {}, {}, {}, {}, {}
    RPEs = []
    
    # Determine nfactor_alpha and nfactor_sigma based on model 
    if setting['model_name'] == "Mackintosh":
        nfactor_alpha = 1
        nfactor_sigma = 0
    elif setting['model_name'] == "Pearce-Hall":
        nfactor_alpha = 0
        nfactor_sigma = 1
    else:
        nfactor_alpha = config["nfactor_alpha"]
        nfactor_sigma = config["nfactor_sigma"]
    
    # Initialize dictionaries for CS
    # Each CS starts with initial V, V_inhib, alpha, and sigma
    values_iter = {key: {} for key in ['other_stimuli', 'odorA', 'odorB']}
    for CS in ['other_stimuli', 'odorA', 'odorB']:
        values_iter[CS]['V_A'] = config["init_V"]
        values_iter[CS]['V_A_inhib'] = config["init_V_inhib"]
        values_iter[CS]['alpha'] = config["init_alpha"]
        values_iter[CS]['sigma'] = config["init_sigma"]
        
        # Calculate initial salience
        values_iter[CS]['salience'] = (config["salience_baseline"] 
                                       + values_iter[CS]['alpha'] * nfactor_alpha 
                                       + values_iter[CS]['sigma'] * nfactor_sigma
                                       )

        # Prepare lists to track the values of V, V_inhib, alpha, sigma, and salience over trials
        V_As[CS] = [config["init_V"]]
        V_A_inhibitories[CS] = [config["init_V_inhib"]]
        alphas[CS] = [config["init_alpha"]]
        sigmas[CS] = [config["init_sigma"]]
        saliences[CS] = [config["salience_baseline"] 
                         + values_iter[CS]['alpha'] * nfactor_alpha 
                         + values_iter[CS]['sigma'] * nfactor_sigma]

    # Main loop over each trial in the simulation
    for n_trial, trial in enumerate(trials[:-1]):
        # Sum V and V_inhib across all CS present in a trial
        V_sum, V_inhib_sum = 0, 0
        for CS in trial['CS']:
            V_sum += values_iter[CS]['V_A']
            V_inhib_sum += values_iter[CS]['V_A_inhib']
        
        # Calculate net associative strength
        V_net = net_associative_strength(V_sum, V_inhib_sum)
        
        # Determine the US strength
        if trial['US'] in ['reward', 'absent', 'timeOut']:
            lambda_ = config["lambda"][trial['US']]
        elif type(trial['US']) == list:
            lambda_ = 'np.nan'
        
        # Compute RPE
        RPE = lambda_ - V_net
        
        # Update the positive and negative associations
        for CS in trial['CS']:
            salience = values_iter[CS]['salience']
            V_A = values_iter[CS]['V_A']
            V_A_inhib = values_iter[CS]['V_A_inhib']
            V_A_net = net_associative_strength(V_A, V_A_inhib)

            V_A, V_A_inhib = update_association(V_A, V_A_inhib, V_A_net, config["lr_v"], 
                                                lambda_, salience, RPE)
            
            # Force 'other_stimuli' associations to be the average of odor A and B associations
            if CS == 'other_stimuli':
                V_A = (values_iter['odorA']['V_A'] + values_iter['odorB']['V_A']) / 2
                V_A_inhib = (values_iter['odorA']['V_A_inhib'] 
                             + values_iter['odorB']['V_A_inhib']) / 2
            
            # Introduce noise in V_A and V_A_inhib
            V_A += np.random.randn() * config["noise_variance_v"]
            V_A_inhib += np.random.randn() * config["noise_variance_v"]
            
            # Save updated V_A and V_A_inhib back to the structure
            values_iter[CS]['V_A'] = V_A
            values_iter[CS]['V_A_inhib'] = V_A_inhib
            
            # Track changes in lists for further analysis
            V_As[CS].append(V_A)
            V_A_inhibitories[CS].append(V_A_inhib)
        
        # Update alpha and sigma / salience
        for CS in trial['CS']:
            # Use the previous V and V_inhib to compute V_A_net and V_X_net
            V_A = V_As[CS][-2]
            V_A_inhib = V_A_inhibitories[CS][-2]
            V_X = V_sum - V_A
            V_X_inhib = V_inhib_sum - V_A_inhib
            
            V_A_net = net_associative_strength(V_A, V_A_inhib)
            V_X_net = net_associative_strength(V_X, V_X_inhib)
            
            # Update alpha and sigma
            alpha = update_alpha(values_iter[CS]['alpha'], 
                                 V_A_net, V_X_net, config["lr_alpha"],
                                 lambda_, config["range_alpha"][CS])
            sigma = update_sigma(values_iter[CS]['sigma'], 
                                 V_net, config["lr_sigma"], 
                                 lambda_, config["range_sigma"])
            
            # Add noise
            alpha += np.random.randn() * config["noise_variance_alpha"]
            sigma += np.random.randn() * config["noise_variance_sigma"]
            
            # Compute salience
            salience = (config["salience_baseline"] 
                        + alpha * nfactor_alpha 
                        + sigma * nfactor_sigma)
            
            # Save the newly computed salience, alpha, and sigma
            values_iter[CS]['salience'] = salience
            values_iter[CS]['alpha'] = alpha
            values_iter[CS]['sigma'] = sigma
            
            # Update the upper limit of alpha
            if config["range_alpha_update"]:
                config["range_alpha"][CS][1] = update_threshold(config["range_alpha"][CS], 
                                                                config["alpha_decey"], 
                                                                lambda_)
            
            # Append updated alpha, sigma, and salience to track them over trials
            alphas[CS].append(alpha)
            sigmas[CS].append(sigma)
            saliences[CS].append(salience)
        
        # For any CS not presented in this trial, we still add noise to stored parameters
        # so each CS has a full time course across trials
        for key in alphas.keys():
            if key not in trial['CS']:
                alpha = alphas[key][-1] + np.random.randn() * config["noise_variance_alpha"]
                sigma = sigmas[key][-1] + np.random.randn() * config["noise_variance_sigma"]
                
                # Recompute salience
                salience = (config["salience_baseline"] 
                            + alpha * nfactor_alpha 
                            + sigma * nfactor_sigma)
                
                # Store the noise-added alpha, sigma, and salience
                alphas[key].append(alpha)
                sigmas[key].append(sigma)
                saliences[key].append(salience)
                
                # Also add noise to V_A and V_A_inhib
                V_As[key].append(V_As[key][-1] + np.random.randn() * config["noise_variance_v"])
                V_A_inhibitories[key].append(V_A_inhibitories[key][-1] 
                                             + np.random.randn() * config["noise_variance_v"])
        
        # Keep track of RPE over trials
        RPEs.append(RPE)
    
    # Return the collected data
    return alphas, sigmas, saliences, RPEs, V_As, V_A_inhibitories

def run_single_simulation(task_name, model_name, nTrials, setting, **kargs):
    """
    Run a single simulation with fixed random seed, specified task, model, and number of trials.
    Additional arguments (kargs) are used to update the default configuration.
    """
    
    # Get the default configuration for the specified model
    configurations = get_configuration(model_name)
    configurations.update(kargs)
    
    # Create or retrieve trials
    realBhvData = setting['realBhvData'] if 'realBhvData' in setting.keys() else False
    trials = get_trials(nTrials, realBhvData=realBhvData, task=task_name)
    
    # Run the learning simulation over the given trials
    alphas, sigmas, saliences, RPEs, V_As, V_A_inhibitories = simulate_learning(trials, configurations, setting)
    
    # Package results into a dictionary
    return {
        'V_As': V_As,
        'RPEs': RPEs,
        'alphas': alphas,
        'sigmas': sigmas,
        'saliences': saliences,
        'V_A_inhibitories': V_A_inhibitories
    }


def run(setting, **kargs):
    """
    Run multiple simulations according to the given setting, optionally in parallel.
    
    """
    # Extract parameters
    nSim, nTrials = setting['nSim'], setting['nTrials']
    task_name, model_name = setting['task_name'], setting['model_name']

    # Run the simulations in parallel
    # results_parallel = Parallel(n_jobs=-1)(
    #     delayed(run_single_simulation)(seed+i, task_name, model_name, nTrials, setting, **kargs) for i in range(nSim)
    # )

    # Run simulations sequentially 
    results_parallel = [
        run_single_simulation(task_name, model_name, nTrials, setting, **kargs) 
        for i in range(nSim)
    ]

    # Initiate a results dictionary for simulations
    results = {
        'V_As': [],
        'RPEs': [],
        'alphas': [],
        'sigmas': [],
        'saliences': [],
        'V_A_inhibitories': []
    }

    # Combine results from each simulation 
    for result in results_parallel:
        for key in results:
            results[key].append(result[key])

    return results

def get_loss(salience_pred, task_name='gonogo'):
    """
    Compute a mean-squared-error (MSE) based loss for salience predictions.

    """
    
    # Select dataset based on the task
    if task_name.lower() == 'reversal':
        data = dataACh_reversal
    if task_name.lower() == 'gonogo':
        data = dataACh_GONOGO

    # Check for NaNs in the predicted salience arrays.
    if np.any(np.isnan(salience_pred['odorA'])) or \
       np.any(np.isnan(salience_pred['odorB'])):
        # Assign a large penalty (loss) if NaNs are present.
        total_loss = 1e6
    else:
        # For the 'reversal' task, compute MSE for odorA (ach_pos) and odorB (ach_neg).
        total_loss  = mean_squared_error(data['ach_pos'], salience_pred['odorA'])
        total_loss += mean_squared_error(data['ach_neg'], salience_pred['odorB'])
    return total_loss


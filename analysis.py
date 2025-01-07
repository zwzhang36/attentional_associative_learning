# -*- coding: utf-8 -*-
"""
Created on May 10th, 2024

@author: Zhewei Zhang

zhzhewei36@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

# In[]
def plot(results, title=None, task='GONOGO'):
    """
    Plot the salience (and related measures) for odorA and odorB across trials.
    """

    # Initialize dictionaries to store arrays of alpha, sigma, and salience data
    alphas_dict = {CS: [] for CS in ['odorA', 'odorB']}
    sigmas_dict = {CS: [] for CS in ['odorA', 'odorB']}
    salience_dict = {CS: [] for CS in ['odorA', 'odorB']}
    
    # Convert lists of dictionaries into numpy arrays for easier processing
    for CS in ['odorA', 'odorB']:
        alphas_dict[CS] = np.array([a[CS] for a in results['alphas']])
        sigmas_dict[CS] = np.array([a[CS] for a in results['sigmas']])
        salience_dict[CS] = np.array([a[CS] for a in results['saliences']])
        
    # Set up a single row of 3 plots for salience, alpha, and sigma
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs = axs.reshape(-1,)  # Flatten the axes array

    # Data categories and y-axis labels
    data_collections = [salience_dict, alphas_dict, sigmas_dict]
    ylabels = [
        'total salience', 
        'predictiveness-driven component, PDS',
        'uncertainty-driven component, UDS'
    ]

    # Loop through each collection (salience, alpha, sigma) and create errorbar plots
    for ith, (data_, ylabel) in enumerate(zip(data_collections, ylabels)):
        for CS in ['odorA', 'odorB']:
            d = data_[CS]
            axs[ith].errorbar(
                range(d.shape[1]),
                np.nanmean(d, axis=0),
                np.nanstd(d, axis=0)/np.sqrt(d.shape[0])
            )

        axs[ith].set_xlabel('trial')
        axs[ith].set_ylabel(ylabel)
        axs[ith].spines['top'].set_visible(False)
        axs[ith].spines['right'].set_visible(False)
        if 'reversal' in task or 'reversal' in title :
            ntrial = d.shape[1]
            axs[ith].set_xlim([ntrial/3-25, ntrial/3+200])
            axs[ith].set_xticks([ntrial/3, ntrial/3+50, ntrial/3+100, ntrial/3+150, ntrial/3+200], 
                                 ['reversal', 50, 100, 150, 200])

    # Set an overall title for the figure
    plt.suptitle(title)


def get_negSalience(results, task='gonogo', nTrials=None, num_pretraining=0):
    """
    Compute and return the negative salience for odorA and odorB after optionally 
    removing pretraining trials. Works for 'gonogo' and 'reversal' tasks.
    """

    salience_dict = {'odorA': [], 'odorB': []}

    if task.lower() == 'gonogo':
        # Extract the salience values for odorA and odorB, skipping pretraining trials
        for CS in ['odorA', 'odorB']:
            salience_dict[CS] = np.array([a[CS][num_pretraining:] for a in results['saliences']])
        
        # Return the mean negative salience across all simulations
        return {
            'odorA': np.nanmean(-salience_dict['odorA'], axis=0),
            'odorB': np.nanmean(-salience_dict['odorB'], axis=0)
        }
    
    if task.lower() == 'reversal':
        # Extract the salience values for odorA and odorB, skipping pretraining trials
        for CS in ['odorA', 'odorB']:            
            salience_dict[CS] = np.array([a[CS][num_pretraining:] for a in results['saliences']])

        # Return the mean negative salience, focusing on the first 2*nTrials data points
        return {
            'odorA': np.nanmean(-salience_dict['odorA'][:, :int(2*nTrials)], axis=0),
            'odorB': np.nanmean(-salience_dict['odorB'][:, :int(2*nTrials)], axis=0)
        }

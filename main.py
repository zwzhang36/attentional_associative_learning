# -*- coding: utf-8 -*-
"""
Created on May 10th, 2024

@author: Zhewei Zhang

zhzhewei36@gmail.com
"""

from scipy.stats import ttest_rel
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt

from models import get_loss
from models import run as run_LePelly
from analysis import plot, get_negSalience


# In[] GONOGO task

# Settings
setting = {'nSim': 50,
           'nTrials': 150,
           'task_name': 'gonogo',
           'model_name': "",  
           }

for model_name, fitted_params in zip(['Mackintosh', 'Pearce-Hall', 'Hybrid'],
                                     [[0.8, 0.3, 0], [0.8, 0., 0.5], [0.8, 0.3, 0.5]]): 
    # Update the model type
    setting.update(model_name = model_name)

    # Run simulation based on fitted parameters and plot 
    results = run_LePelly(setting, 
                    lr_v =      fitted_params[0],
                    lr_alpha = {k:fitted_params[1] for k in ['up', 'down']},
                    lr_sigma = {k:fitted_params[2] for k in ['up', 'down']},
                    )
    
    # Plot the salience dynamics
    plot(results, title=setting['task_name']+'_'+setting['model_name'])
    
    # Save figures
    # plt.savefig(setting['task_name']+'_'+setting['model_name']+'neg_salience.eps')
    plt.show()

# In[] For the GONOGO task fitting

loss_dict = {k:[] for k in ['Mackintosh', 'Pearce-Hall', 'Hybrid']}
for model in loss_dict.keys():
    setting = {'nSim': 3,
               'nTrials': 150,
               'task_name': 'gonogo',
               'realBhvData': True,
               'model_name': model,
               }
    
    # Define the hyperparameter search space in log scale for stability
    param_space = [
        Real(-2.0, 0.0, name='log_p1'),      # Logarithmic range for lr_v
        Real(-3.0, 0.0, name='log_p2'),      # Logarithmic range for lr_alpha for Mackintosh / Hybrid
        Real(-3.0, 0.0, name='log_p3'),      # Logarithmic range for lr_sigma
    ]
    
    @use_named_args(param_space)
    def get_loss_gonogo(log_p1, log_p2, log_p3):
        # Convert log-scale hyperparameters back to normal scale
        p1 = 10 ** log_p1
        p2 = 10 ** log_p2
        p3 = 10 ** log_p3

        # Run simulations in parallel to improve efficiency
        if model == 'Mackintosh':
            results = run_LePelly(setting, 
                          lr_v  =    p1,                   # Learning rate for learning assocaitions
                          lr_alpha = {'up':p2, 'down':p3}, # Learning rate for changes in alpha
                          lr_sigma = {'up':0,  'down':0},  # Learning rate for changes in sigma
                          )
        elif model == 'Pearce-Hall':
            results = run_LePelly(setting, 
                          lr_v  =     p1,                  # Learning rate for learning assocaitions
                          lr_alpha = {'up':0, 'down':0},   # Learning rate for changes in alpha
                          lr_sigma = {'up':p2, 'down':p3}, # Learning rate for changes in sigma
                          )
        elif model == 'Hybrid':
            results = run_LePelly(setting, 
                          lr_v  =    p1,                   # Learning rate for learning assocaitions
                          lr_alpha = {'up':p2, 'down':p2}, # Learning rate for changes in alpha
                          lr_sigma = {'up':p3, 'down':p3}, # Learning rate for changes in sigma
                          )
    
        # Calculate average negative salience across all simulations and compute average loss
        salience_pred = get_negSalience(results, task=setting['task_name'], 
                                        nTrials = setting['nTrials'])
    
        # Check if salience_pred contains NaNs before calculating the loss
        loss = get_loss(salience_pred, task_name=setting['task_name'])/setting['nSim']
        return loss
        
    # Bounds for differential evolution (same as numeric_bounds)
    diff_ev_bounds = [(-2.0, 0.0), (-3.0, 0.0), (-3.0, 0.0)]
    
    res_diff_ev = differential_evolution(get_loss_gonogo, bounds=diff_ev_bounds, strategy='best1bin', maxiter=100, popsize=15)
    loss_dict[model] = res_diff_ev.population_energies


for model, loss in loss_dict.items():
    print("{} - average loss:{}".format(model, loss.mean()))

print('compare the goodness of fitiing')
print('     Pearce-Hall vs Hybrid models:  ', 
      ttest_rel(loss_dict['Pearce-Hall'], loss_dict['Hybrid']))
print('     Mackintosh vs Hybrid models:  ', 
      ttest_rel(loss_dict['Mackintosh'], loss_dict['Hybrid']))


# In[] Simulating the Reversal Learning Task

# Configure simulation settings in a dictionary
setting = {
    'nSim': 50,        
    'nTrials': 150,   
    'task_name': 'reversal',  
    'model_name': "Hybrid",   
}

# Run the simulation using run_LePelly,
results = run_LePelly(
    setting,
    lr_v=0.8,                           # Learning rate for V_A / V_A_inhib
    lr_alpha={'up': 0.3, 'down': 0.5},  # Learning rate for alpha 
    lr_sigma={'up': 0.15, 'down': 0.5}, # Learning rate for sigma 
)

# Plot the salience.
plot(results, title=setting['task_name'] + '_' + setting['model_name'])

# Save the figure to a file
# plt.savefig(setting['task_name'] + '_' + setting['model_name'] + '_salience.png')

# Display
plt.show()

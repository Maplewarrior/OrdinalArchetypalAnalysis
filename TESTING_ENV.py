from OAA_class_adj import _OAA
from RBOAA_class_adj import _RBOAA

from OAA_class import _OAA as _OAA_old
from RBOAA_class import _RBOAA as _RBOAA_old
from synthetic_data_class import _synthetic_data

import matplotlib.pyplot as plt
import numpy as np
from eval_measures import NMI, MCC


""" Define parameters:
    - N:         Respondents, M = questions
    - K:         Number of archetypes
    - p:         Number of points on likert scale in synthetic data 
    - a_param:   Affects the weighting of archetypes on respondents
    - b_param:   Affects the response bias in the synthetic data, low value --> high RB
    - sigma:     Noise parameter, high value --> high noise
    - sigma_dev: Determines the variation in sigma when modelled on each individual
    - rb:        Boolean, whether to have response bias in synthetic data  """
    
N, M = 2500, 12
K = 3
p = 6
a_param = 1
b_param = 5
sigma = -2.2
sigmas = [-2.2, -1.5, -1.05, -0.4002] 
sigma_dev = 0.0
rb = True
n_iter = 5000
mute=True
lr = 0.01


def compareApproaches(n_repeats):
    
    OAA = _OAA()
    RBOAA = _RBOAA()
    RBOAA_old = _RBOAA_old()
    
    RB_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}
    
    RB_old_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}
    
    OAA_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}

    
    for idx, s in enumerate(sigmas):
        print("iter no:", idx)
        
        syn = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std=sigma_dev)
        columns = syn.columns
        
        for i in range(n_repeats):
            
            ### Compute NMI/MCC for alternating RBOAA ###
            RB = RBOAA._compute_archetypes_alternating(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True)
            RB_res['NMI'][i][idx] = NMI(syn.A, RB.A)
            RB_res['MCC'][i][idx] = MCC(syn.Z, RB.Z)
            
            #### Compute NMI/MCC for old RBOAA ###
            RB_old = RBOAA_old._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_synthetic_data=True, with_OAA_initialization=True, early_stopping=True)
            RB_old_res['NMI'][i][idx] = NMI(syn.A, RB_old.A)
            RB_old_res['MCC'][i][idx] = MCC(syn.Z, RB_old.Z)
            
            ### Compute NMI/MCC for OAA ###
            O =  OAA._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, early_stopping=True)
            OAA_res['NMI'][i][idx] = NMI(syn.A, O.A)
            OAA_res['NMI'][i][idx]= MCC(syn.Z, O.Z)
            
            
    return RB_res, RB_old_res, OAA_res
            
            
            
            
def plot_results(sigmas, result, title):
    
    sigmas_c = [np.log(1+np.exp(s)) for s in sigmas]
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    
    
    
    NMIs = np.mean(result['NMI'],axis=0)
    MCCs = np.mean(result['MCC'], axis=0)
    ax[0].plot(sigmas_c, NMIs)
    ax[0].set_title('NMI plot')
    ax[0].set_ylabel('NMI')
    
      
    # ax[0].boxplot(result['NMI'], positions=sigmas_c, notch=True)
    
    
    ax[1].plot(sigmas_c, MCCs)
    ax[1].set_title('MCC plot')
    ax[1].set_ylabel('MCC')
    
    # ax[1].boxplot(result['MCC'], positions=sigmas_c, notch=True)
    
    # ax[1].title('MCC plot')
    fig.supxlabel('true sigma')
    # ax[0].xticks(locs)
    plt.show()
    
    
    
            
#%%
RB_r, RB_old_r, OAA_r = compareApproaches(5)






    
#%%
# print("1st plot = alternating RBOAA")
# print("2nd plot = Old RBOAA")
# print("3rd plot = OAA")


# dummy = {'NMI':np.array([[0.9, 0.85, 0.82, 0.85],[0.65, 0.54, 0.45, 0.14]]),
#          'MCC':np.array([[0.9, 0.83, 0.82, 0.55],[0.76, 0.65, 0.67,0.55]])}

# plot_results(sigmas, dummy, "dummy")
# print(np.mean(dummy['NMI'], axis=0))
#%%

plot_results(sigmas, RB_r)
plot_results(sigmas, RB_old_r)
plot_results(sigmas, OAA_r)


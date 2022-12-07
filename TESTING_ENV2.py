from OAA_class import _OAA
from RBOAA_class import _RBOAA
from RBOAA_class_old import _RBOAA as _RBOAA_old
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
K = 5
p = 6
a_param = 0.85
b_param = 5
#sigmas = [-2.2, -1.5, -1.05, -0.4002] 
sigmas = [-1.5,-1.05] 
sigma_dev = 1
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

    RB_old_script_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}
    
    OAA_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}

    OAA_old_res = {'NMI': np.empty((n_repeats, len(sigmas))), 
              'MCC': np.empty((n_repeats, len(sigmas)))}


    
    for idx, s in enumerate(sigmas):
        print("sigma no:", idx)
        
        syn = _synthetic_data(N, M, K, p, s, rb, a_param, b_param, sigma_std=sigma_dev)
        columns = syn.columns
        
        for i in range(n_repeats):
            print("rep no:", i)
            
            ### Compute NMI/MCC for alternating RBOAA ###
            RB = RBOAA._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True, alternating=True)
            RB_res['NMI'][i][idx] = NMI(syn.A, RB.A)
            RB_res['MCC'][i][idx] = MCC(syn.Z, RB.Z)

            ### Compute NMI/MCC for alternating RBOAA ###
            RB_old_script = RBOAA_old._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True)
            RB_old_script_res['NMI'][i][idx] = NMI(syn.A, RB_old_script.A)
            RB_old_script_res['MCC'][i][idx] = MCC(syn.Z, RB_old_script.Z)

            #### Compute NMI/MCC for old RBOAA ###
            #RB_old = RBOAA._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True, alternating=False)
            #RB_old_res['NMI'][i][idx] = NMI(syn.A, RB_old.A)
            #RB_old_res['MCC'][i][idx] = MCC(syn.Z, RB_old.Z)
            
            ### Compute NMI/MCC for OAA ###
            #O =  OAA._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, early_stopping=True, alternating=True)
            #OAA_res['NMI'][i][idx] = NMI(syn.A, O.A)
            #OAA_res['MCC'][i][idx]= MCC(syn.Z, O.Z)

            ### Compute NMI/MCC for OAA ###
            #O_old =  OAA._compute_archetypes(syn.X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, early_stopping=True,alternating=False)
            #OAA_old_res['NMI'][i][idx] = NMI(syn.A, O_old.A)
            #OAA_old_res['MCC'][i][idx]= MCC(syn.Z, O_old.Z)
            
            
    return RB_res, RB_old_res, RB_old_script_res, OAA_res, OAA_old_res
            
            
            
            
def plot_results(sigmas, result):
    
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
RB_r, RB_old_r, RB_old_script_res, OAA_r, OAA_old_r = compareApproaches(2)






    
#%%
# print("1st plot = alternating RBOAA")
# print("2nd plot = Old RBOAA")
# print("3rd plot = OAA")


# dummy = {'NMI':np.array([[0.9, 0.85, 0.82, 0.85],[0.65, 0.54, 0.45, 0.14]]),
#          'MCC':np.array([[0.9, 0.83, 0.82, 0.55],[0.76, 0.65, 0.67,0.55]])}

# plot_results(sigmas, dummy, "dummy")
# print(np.mean(dummy['NMI'], axis=0))
#%%

print("RBOAA NEW")
print(RB_r)
print("RBOAA OLD SCRIPT")
print(RB_old_script_res)
print("RBOAA OLD")
print(RB_old_r)
print("OAA NEW")
print(OAA_r)
print("OAA OLD")
print(OAA_old_r)

plot_results(sigmas, RB_r)
plot_results(sigmas, RB_old_script_res)
plot_results(sigmas, RB_old_r)
plot_results(sigmas, OAA_r)
plot_results(sigmas, OAA_old_r)


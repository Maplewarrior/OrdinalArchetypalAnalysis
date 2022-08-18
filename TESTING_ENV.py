from OAA_class_adj import _OAA
from RBOAA_class_adj import _RBOAA

from OAA_class import _OAA as _OAA_old
from RBOAA_class import _RBOAA as _RBOAA_old
from synthetic_data_class import _synthetic_data

import matplotlib.pyplot as plt
import numpy as np
from eval_measures import NMI, MCC

N, M = 10000, 15
K = 3
p = 5
a_param = 1
b_param = 5
sigma = -8
sigmas = [-10, -5, -3, -1]

sigma_dev = 0.25
rb = True
n_iter = 5000
mute=False

lr = 0.01

syn = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std=sigma_dev)
columns = syn.columns

OAA = _OAA()
RBOAA = _RBOAA()
OAA_old = _OAA_old()
RBOAA_old = _RBOAA_old()

NMIs = []
MCCs = []

NMI_old = []
MCC_old = []

#%%
# RB = RBOAA._compute_archetypes_alternating(syn.X, K, p, n_iter, lr, mute, columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True)
# print("Alternating archetypal analysis")
# print("NMI: ", NMI(syn.A, RB.A))
# print("MCC: ", MCC(syn.Z, RB.Z))

# RB_old = RBOAA_old._compute_archetypes(syn.X, K, p, n_iter, lr, mute, columns, with_synthetic_data=True, with_OAA_initialization=True, early_stopping=True)
# print("NMI old: ", NMI(syn.A, RB_old.A))
# print("MCC old: ", MCC(syn.Z, RB_old.Z))

# print("Old archetypal analysis:")
# result_old = OAA_old._compute_archetypes(syn.X, K, p, lr=0.01, n_iter=n_iter, mute=mute, columns=columns, early_stopping=True)


#%%
reps = 3
NMIs = np.empty((reps, len(sigmas)))
MCCs = np.empty((reps, len(sigmas)))

NMI_old = np.empty((reps, len(sigmas)))
MCC_old = np.empty((reps, len(sigmas)))


sigmas_c = [np.log(1+np.exp(s)) for s in sigmas]


for i, sigma in enumerate(sigmas):
    syn = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std=sigma_dev)
    columns = syn.columns
    for rep in range(reps):

        # result_alt = OAA._compute_archetypes_alternating(syn.X, K, p, n_iter, lr, mute, columns, sigma_cap=False, early_stopping=True) 
        
        RB = RBOAA._compute_archetypes_alternating(syn.X, K, p, n_iter, lr, mute, columns, with_OAA_initialization=True, early_stopping=True, with_synthetic_data = True)
        NMIs[rep,i] = NMI(syn.A, RB.A)
        MCCs[rep,i] = MCC(syn.Z, RB.Z)
        
        # result_old = OAA_old._compute_archetypes(syn.X, K, p, lr=0.01, n_iter=n_iter, mute=mute, columns=columns, early_stopping=True)
        RB_old = RBOAA_old._compute_archetypes(syn.X, K, p, n_iter, lr, mute, columns, with_synthetic_data=True, with_OAA_initialization=False, early_stopping=True)
        
        NMI_old[rep,i] = NMI(syn.A, RB_old.A)
        MCC_old[rep,i] = MCC(syn.Z, RB_old.Z)
    
    
# print("Alternating archetypal analysis")
# print("NMI: ", NMI(syn.A, result_alt.A))
# print("MCC: ", MCC(syn.Z, result_alt.Z))

# print("Old archetypal analysis:")
# result_old = OAA_old._compute_archetypes(syn.X, K, p, lr=0.01, n_iter=n_iter, mute=mute, columns=columns, early_stopping=True)
# print("NMI old: ", NMI(syn.A, result_old.A))
# print("MCC old: ", MCC(syn.Z, result_old.Z))


#%%

print("ALTERNATING ANALYSIS:")
print("NMI:\n")
print(NMIs)
print("MCC:\n")
print(MCCs)


print("OLD OAA:")
print("NMI")
print(NMI_old)
print("MCC\n", MCC_old)
# result = OAA._compute_archetypes(syn.X, K, p, n_iter, lr, mute, columns, sigma_cap=False, early_stopping=True) 

# print(result.loss)
print("sigma vals: \n", sigmas_c)

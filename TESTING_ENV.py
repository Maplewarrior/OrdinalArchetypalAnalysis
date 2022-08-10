from OAA_class_adj import _OAA
from synthetic_data_class import _synthetic_data
N, M = 5000, 10
K = 3
p = 5
a_param = 1
b_param = 10
sigma = -8
sigma_dev = 0.0
rb = True
n_iter = 2000
mute=False

lr = 0.01

syn = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std=sigma_dev)
columns = syn.columns

OAA = _OAA()


#%% OAA

result_alt = OAA._compute_archetypes_alternating(syn.X, K, p, n_iter, lr, mute, columns, sigma_cap=False, early_stopping=True) 

#%%
result = OAA._compute_archetypes(syn.X, K, p, n_iter, lr, mute, columns, sigma_cap=False, early_stopping=True) 

# print(result.loss)



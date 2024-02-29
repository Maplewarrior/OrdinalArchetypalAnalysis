from src.AAM import AA as AA_class
from src.utils.eval_measures import NMI, MCC
import numpy as np
import pdb
from src.methods.CAA_class import _CAA
from src.methods.OAA_class import _OAA
from src.methods.RBOAA_class import _RBOAA
from src.utils.eval_measures import NMI, MCC

from src.methods.OAA_class_old import _OAA as _OAA_old
from src.methods.RBOAA_class_old import _RBOAA as _RBOAA_old

"""
#### Findings ####
Synthetic data:
    - rb = False --> CAA better than OAA better than RBOAA
    - rb = True & b_param <= 5 --> RBOAA better
    - Increase questions --> increase NMI
    - rb = True + sigma_dev --> new method might improved on old RBOAA.

OAA:    
    - alternating decreases performance...
    - no significant difference otherwise
    - Sometimes NaN losses occur --> only when beta_regulators = True?
    - Likely a mistake somewhere --> when doing RBOAA_new with OAA_new hotstart things mess up whereas without hotstart results are comparable.
        - OAA old and new OAA have the same performance...
RBOAA:



"""

### define hyperparameters for synthetic data and create a dataset
N, M = 500, 12
K = 4
p = 6
a_param = 0.85
b_param = 3
sigma = -3.0
sigma_dev = 0.05
rb = True
n_iter = 20000
CAA_lr = 0.01
OAA_lr = 0.001
mute = True
early_stopping = True

# create dataset
AA = AA_class()
AA.create_synthetic_data(N, M, K, p, sigma, rb, b_param, a_param, sigma_dev,mute=mute)
columns = AA._synthetic_data.columns

_X = AA._synthetic_data.X # extract synthetic responses
_Z = AA._synthetic_data.Z
_A = AA._synthetic_data.A

CAA = _CAA()
CAA_res = CAA._compute_archetypes(X=_X, K=K, p=p, n_iter=n_iter, lr=CAA_lr, mute=mute, 
                        early_stopping=early_stopping, columns=columns, with_synthetic_data=True)

print(f'CAA NMI: {NMI(CAA_res.A, _A)}')
print(f'CAA MCC: {MCC(CAA_res.Z, _Z)}\n')
"""
OAA variants:
- Just OAA
- With CAA init
- With beta regulators
- With alternating start --> Keep sigma fixed
    - Any combination of the 3 previous.
"""

### Old OAA
# OAA_old = _OAA_old()
# OAA_res = OAA_old._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, columns=columns, early_stopping=early_stopping)
# print(f'OAA old NMI: {NMI(OAA_res.A, _A)}')
# print(f'OAA old MCC: {MCC(OAA_res.Z, _Z)}\n')
### Just OAA
# OAA = _OAA()
# OAA_res = OAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, with_CAA_initialization=False,columns=columns, beta_regulators=False, early_stopping=early_stopping)
# print(f'OAA NMI: {NMI(OAA_res.A, _A)}')
# print(f'OAA MCC: {MCC(OAA_res.Z, _Z)}\n')

### CAA init
# OAA = _OAA()
# OAA_res = OAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, with_CAA_initialization=True,columns=columns, beta_regulators=False, early_stopping=early_stopping)
# print(f'OAA (init) NMI: {NMI(OAA_res.A, _A)}')
# print(f'OAA (init) MCC: {MCC(OAA_res.Z, _Z)}')

### Beta Reg
# OAA = _OAA()
# OAA_res = OAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, with_CAA_initialization=False,columns=columns, beta_regulators=True, early_stopping=early_stopping, backup_itterations=True)
# print(f'OAA (reg) NMI: {NMI(OAA_res.A, _A)}')
# print(f'OAA (reg) MCC: {MCC(OAA_res.Z, _Z)}\n')

### alternating + Reg
# OAA = _OAA()
# OAA_res = OAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, with_CAA_initialization=False, columns=columns, beta_regulators=True, alternating=True, early_stopping=early_stopping)
# print(f'OAA (alt + reg) NMI: {NMI(OAA_res.A, _A)}')
# print(f'OAA (alt + reg) MCC: {MCC(OAA_res.Z, _Z)}\n')


"""
RBOAA variants
"""
### Regular
RBOAA = _RBOAA()
RBOAA_res = RBOAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, columns, True, True, with_OAA_initialization=True, beta_regulators=False, backup_itterations=True)
print(f'RBOAA NMI: {NMI(RBOAA_res.A, _A)}')
print(f'RBOAA MCC: {MCC(RBOAA_res.Z, _Z)}\n')

### old implementation
RBOAA_old = _RBOAA_old()
RBOAA_res = RBOAA_old._compute_archetypes(X=_X, K=K, p=p, n_iter=n_iter, lr=OAA_lr, mute=mute, columns=columns, with_synthetic_data=True, early_stopping=True, with_OAA_initialization=True)
print(f'RBOAA old NMI: {NMI(RBOAA_res.A, _A)}')
print(f'RBOAA old MCC: {MCC(RBOAA_res.Z, _Z)}\n')


# RBOAA = _RBOAA()
# RBOAA_res = RBOAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, columns, True, True, with_OAA_initialization=True, beta_regulators=True, backup_itterations=True)
# print(f'RBOAA (b_reg + hotstart) NMI: {NMI(RBOAA_res.A, _A)}')
# print(f'RBOAA (b_reg + hotstart) MCC: {MCC(RBOAA_res.Z, _Z)}\n')


# RBOAA = _RBOAA()
# RBOAA_res = RBOAA._compute_archetypes(_X, K, p, n_iter, OAA_lr, mute, columns, True, True, with_OAA_initialization=False, beta_regulators=True, backup_itterations=True, alternating=True)
# print(f'RBOAA (b_reg + alternating) NMI: {NMI(RBOAA_res.A, _A)}')
# print(f'RBOAA (b_reg + alternating) MCC: {MCC(RBOAA_res.Z, _Z)}\n')
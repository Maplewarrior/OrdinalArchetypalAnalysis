from src.methods.RBOAA_class_old import _RBOAA
from src.methods.OAA_class_old import _OAA
from src.AAM import AA as AA_class
from src.utils.eval_measures import NMI, MCC
import numpy as np

N, M = 5000, 10
K = 5
p = 6
a_param = 0.85
b_param = 5
sigma = -3.0
sigma_dev = 0.0
rb = False
n_iter = 10000
lr = 0.01
mute = True
early_stopping = True

results = {"regular":[],"alternating":[],"beta":[],"backup":[],"all":[]}

for i in range(5):
    print("ITT: ", i)
    AA = AA_class()
    AA.create_synthetic_data(N, M, K, p, sigma, rb, b_param, a_param, sigma_dev,mute=mute)

    AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, beta_regulators=False, itteration_backup=False, mute=mute)
    print("NMI OAA:", NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    results["regular"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    #AA.plot("OAA","loss_plot",with_synthetic_data=True,mute=mute)

    AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=True, beta_regulators=False, itteration_backup=False, mute=mute)
    print("NMI OAA ALT.:", NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    results["alternating"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    #AA.plot("OAA","loss_plot",with_synthetic_data=True,mute=mute)

    AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, beta_regulators=True, itteration_backup=False, mute=mute)
    print("NMI OAA BETA:", NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    results["beta"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    #AA.plot("OAA","loss_plot",with_synthetic_data=True,mute=mute)

    AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, beta_regulators=False, itteration_backup=True, mute=mute)
    print("NMI OAA BACKUP:", NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    results["backup"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    #AA.plot("OAA","loss_plot",with_synthetic_data=True,mute=mute)

    AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=True, mute=mute,beta_regulators=True,itteration_backup=True)
    print("NMI OAA ALL:", NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    results["all"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["OAA"][0].A))
    #AA.plot("OAA","loss_plot",with_synthetic_data=True,mute=mute)

print("FINAL")
print("regular: ", np.mean(results["regular"]))
print("alternating: ", np.mean(results["alternating"]))
print("beta: ", np.mean(results["beta"]))
print("backup: ", np.mean(results["backup"]))
print("all: ", np.mean(results["all"]))



# results = {"regular":[],"alternating":[],"beta":[],"backup":[],"all":[]}

# for i in range(5):
#     print("ITT: ", i)

#     AA = AA_class()
#     AA.create_synthetic_data(N, M, K, p, sigma, rb, b_param, a_param, sigma_dev,mute=mute)
    
#     AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="RBOAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, mute=mute,beta_regulators=False,itteration_backup=False)
#     print("NMI RBOAA:", NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     results["regular"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     # AA.plot("RBOAA","loss_plot",with_synthetic_data=True)

#     AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="RBOAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=True, mute=mute,beta_regulators=False,itteration_backup=False)
#     print("NMI RBOAA ALT.:", NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     results["alternating"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     #AA.plot("RBOAA","loss_plot",with_synthetic_data=True)

#     AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="RBOAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, mute=mute, beta_regulators=True,itteration_backup=False)
#     print("NMI RBOAA BETA:", NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     results["beta"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     # AA.plot("RBOAA","loss_plot",with_synthetic_data=True)

#     AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="RBOAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=False, mute=mute,beta_regulators=False,itteration_backup=True)
#     print("NMI RBOAA BACKUP:", NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     results["backup"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     # AA.plot("RBOAA","loss_plot",with_synthetic_data=True)

#     AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=early_stopping, model_type="RBOAA", lr=lr, with_synthetic_data=True, with_hot_start=True, alternating=True, mute=mute,beta_regulators=True,itteration_backup=True)
#     print("NMI RBOAA ALL:", NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     results["all"].append(NMI(AA._synthetic_data.A,AA._synthetic_results["RBOAA"][0].A))
#     # AA.plot("RBOAA","loss_plot",with_synthetic_data=True)¨

# print("FINAL")
# print("regular: ", np.mean(results["regular"]))
# print("alternating: ", np.mean(results["alternating"]))
# print("beta: ", np.mean(results["beta"]))
# print("backup: ", np.mean(results["backup"]))
# print("all: ", np.mean(results["all"]))
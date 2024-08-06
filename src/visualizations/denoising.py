
import numpy as np
import matplotlib.pyplot as plt
from src.visualizations.functions import findProb

def denoising(X, X_cor, dataObjCorr, K_list, p: int, n_reps: int, my_pallette: dict, savedir: str,):

    idx = np.nonzero(X-X_cor)

    RMSE_CAA = np.zeros((len(K_list),n_reps))
    RMSE_RBOAA = np.zeros((len(K_list),n_reps))
    RMSE_OAA = np.zeros((len(K_list),n_reps))
    RMSE_TSAA = np.zeros((len(K_list),n_reps))

    R_corr_CAA = np.zeros((len(K_list),n_reps),dtype= object)
    R_corr_RBOAA = np.zeros((len(K_list),n_reps),dtype= object)
    R_corr_OAA = np.zeros((len(K_list),n_reps),dtype= object)
    R_corr_TSAA = np.zeros((len(K_list),n_reps),dtype= object)

    for i in range(len(K_list)): 
        for j in range(n_reps):

            k = K_list[i]

            R_estRBOAA = findProb(dataObjCorr,'RBOAA', k, j, p)
            R_estOAA = findProb(dataObjCorr,'OAA', k, j, p)


            R_corr_CAA[i,j] = X @ dataObjCorr['CAA'][f'K{k}'][j].B @ dataObjCorr['CAA'][f'K{k}'][j].A
            R_corr_OAA[i,j] = R_estOAA.numpy() 
            R_corr_RBOAA[i,j] = R_estRBOAA.numpy() 
            R_corr_TSAA[i,j] = X @ dataObjCorr['TSAA'][f'K{k}'][j].B @ dataObjCorr['TSAA'][f'K{k}'][j].A

    for i in range(len(K_list)):
        for j in range(n_reps):
            RMSE_CAA[i,j] = np.sqrt(((X[idx]- R_corr_CAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_OAA[i,j] = np.sqrt(((X[idx]- R_corr_OAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_RBOAA[i,j] = np.sqrt(((X[idx]- R_corr_RBOAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_TSAA[i,j] = np.sqrt(((X[idx]- R_corr_TSAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))

    fig, ax = plt.subplots(1,1, figsize = (15,5), layout='constrained')
    ax.plot(range(len(K_list)), RMSE_CAA, c = my_pallette['AA'],alpha = 0.5)
    ax.plot(range(len(K_list)), RMSE_RBOAA,c = my_pallette['RBOAA'],alpha= 0.5)
    ax.plot(range(len(K_list)), RMSE_OAA,c = my_pallette['OAA'],alpha= 0.5)
    ax.plot(range(len(K_list)), RMSE_TSAA,c = my_pallette['TSAA'],alpha = 0.5)

    ax.plot(range(len(K_list)), np.min(RMSE_CAA,axis = 1), c = my_pallette['AA'],label = 'AA')
    ax.plot(range(len(K_list)), np.min(RMSE_RBOAA,axis = 1),c =  my_pallette['RBOAA'],label = 'RBOAA')
    ax.plot(range(len(K_list)), np.min(RMSE_OAA,axis = 1),c =  my_pallette['OAA'],label = 'OAA')
    ax.plot(range(len(K_list)), np.min(RMSE_TSAA,axis = 1),c =  my_pallette['TSAA'],label = 'TSAA')

    ax.set_xlabel("Number of Archetypes", fontsize = 30)
    ax.set_ylabel("RMSE", fontsize = 30)

    ax.set_xticks(range(len(K_list)))
    ax.set_xticklabels(K_list)

    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(loc='upper right',fontsize = 30)

    plt.savefig(f"{savedir}/denoising_error_plot.png", dpi=1000)


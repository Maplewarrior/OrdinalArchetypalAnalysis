
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from src.visualizations.functions import findProb


my_pallette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSOAA" : "#073B4C","GT": "#9055A2"}


def denoising(dataObj,dataObjCorr,dataObjOSMCorr,K_list,p,figName):

    X = dataObj['CAA']['K1'][3].X
    X_cor = dataObjCorr['CAA']['K1'][3].X
    idx = np.nonzero(X-X_cor)

    RMSE_CAA = np.zeros((len(K_list),10))
    RMSE_RBOAA = np.zeros((len(K_list),10))
    RMSE_OAA = np.zeros((len(K_list),10))
    RMSE_TSOAA = np.zeros((len(K_list),10))

    R_corr_CAA = np.zeros((len(K_list),10),dtype= object)
    R_corr_RBOAA = np.zeros((len(K_list),10),dtype= object)
    R_corr_OAA = np.zeros((len(K_list),10),dtype= object)
    R_corr_TSOAA = np.zeros((len(K_list),10),dtype= object)



    for i in range(len(K_list)): 
        for j in range(10):

            k = K_list[i]

            R_estRBOAA = findProb(dataObjCorr,'RBOAA', k, j, p)
            R_estOAA = findProb(dataObjCorr,'OAA', k, j, p)


            R_corr_CAA[i,j] = dataObj['CAA']['K1'][3].X@dataObjCorr['CAA'][f'K{k}'][j].B@dataObjCorr['CAA'][f'K{k}'][j].A
            R_corr_OAA[i,j] = R_estOAA.numpy() 
            R_corr_RBOAA[i,j] = R_estRBOAA.numpy() 
            R_corr_TSOAA[i,j] = dataObj['CAA']['K1'][3].X@dataObjOSMCorr['TSAA'][f'K{k}'][j].B@dataObjOSMCorr['TSAA'][f'K{k}'][j].A

    for i in range(len(K_list)):
        for j in range(10):
            RMSE_CAA[i,j] = np.sqrt(((X[idx]- R_corr_CAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_OAA[i,j] = np.sqrt(((X[idx]- R_corr_OAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_RBOAA[i,j] = np.sqrt(((X[idx]- R_corr_RBOAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))
            RMSE_TSOAA[i,j] = np.sqrt(((X[idx]- R_corr_TSOAA[i,j][idx])**2).sum())/np.sqrt(len(idx[1]))


    fig, ax = plt.subplots(1,1, figsize = (15,5), layout='constrained')
    ax.plot(range(len(K_list)), RMSE_CAA, c = my_pallette['AA'],alpha = 0.5)
    ax.plot(range(len(K_list)), RMSE_RBOAA,c = my_pallette['RBOAA'],alpha= 0.5)
    ax.plot(range(len(K_list)), RMSE_OAA,c = my_pallette['OAA'],alpha= 0.5)
    ax.plot(range(len(K_list)), RMSE_TSOAA,c = my_pallette['TSOAA'],alpha = 0.5)

    ax.plot(range(len(K_list)), np.min(RMSE_CAA,axis = 1), c = my_pallette['AA'],label = 'AA')
    ax.plot(range(len(K_list)), np.min(RMSE_RBOAA,axis = 1),c =  my_pallette['RBOAA'],label = 'RBOAA')
    ax.plot(range(len(K_list)), np.min(RMSE_OAA,axis = 1),c =  my_pallette['OAA'],label = 'OAA')
    ax.plot(range(len(K_list)), np.min(RMSE_TSOAA,axis = 1),c =  my_pallette['TSOAA'],label = 'TSAA')


    ax.set_xlabel("Number of Archetypes", fontsize = 30)
    ax.set_ylabel("RMSE", fontsize = 30)

    ax.set_xticks(range(len(K_list)))
    ax.set_xticklabels(K_list)

    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(loc='upper right',fontsize = 30)

    plt.savefig("Plots_for_paper/"+figName+".png",dpi=1000)


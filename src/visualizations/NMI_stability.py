import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
import os
from src.utils.eval_measures import NMI
from src.visualizations.functions import load_result_obj

def calcMI(z1,z2):
    eps = 10e-16
    P = z1@z2.T
    PXY = P/P.sum()
    PXPY = np.outer(np.expand_dims(PXY.sum(1), axis=0),np.expand_dims(PXY.sum(0), axis=1))
    ind = np.nonzero(PXY>0)
    MI = np.sum(PXY*np.log(eps+PXY/(eps+PXPY)))
    return MI
    
def calcNMI(z1,z2):
    NMI=(2*calcMI(z1,z2))/(calcMI(z1,z1)+calcMI(z2,z2))
    #NMI = NMI.reshape((z1.shape[0], z1.shape[0]))
    
    return NMI

def plot_NMI_stability(folder_path: str, K_list: list[int], repetitions: int, methods_colors: dict = None, savedir: str = None):
    
    methods = ['RBOAA', 'OAA', 'CAA', 'TSAA']
    test = itertools.combinations(range(repetitions), 2)
    t = list(test)
    calcIDX = np.array(t)

    NMI_RBOAA_complex_large = np.zeros((len(K_list),len(calcIDX)))
    NMI_OAA_complex_large = np.zeros((len(K_list),len(calcIDX)))
    NMI_AA_complex_large = np.zeros((len(K_list),len(calcIDX)))
    NMI_TSAA_complex_large = np.zeros((len(K_list),len(calcIDX)))

    for method in methods:
        i = 0
        for K in K_list:
            for j in range(len(calcIDX)):
                # if method == "TSAA":
                #     if 'TSAA_objects' in os.listdir(folder_path):
                #         filename1 = f'{folder_path}/TSAA_objects/CAA_K={str(K)}_rep={str(calcIDX[j,0])}' 
                #         filename2 = f'{folder_path}/TSAA_objects/CAA_K={str(K)}_rep={str(calcIDX[j,1])}'
                #         file1 = load_result_obj(filename1)
                #         file2 = load_result_obj(filename2)
                #         file1 = file1.A
                #         file2 = file2.A
                #     else:
                #         continue
                # eslse:
                f'{folder_path}/{method}_objects/{method}_K={str(K)}_rep={str(calcIDX[j,0])}'
                filename1 = f'{folder_path}/{method}_objects/{method}_K={str(K)}_rep={str(calcIDX[j,0])}'
                filename2 = f'{folder_path}/{method}_objects/{method}_K={str(K)}_rep={str(calcIDX[j,1])}'
                file1 = load_result_obj(filename1)
                file2 = load_result_obj(filename2)
                file1 = file1.A
                file2 = file2.A

                if method == "RBOAA":
                    NMI_RBOAA_complex_large[i,j] = calcNMI(file1,file2)

                elif method == "OAA":
                    NMI_OAA_complex_large[i,j] = calcNMI(file1,file2)

                elif method == "CAA":
                    NMI_AA_complex_large[i,j] = calcNMI(file1,file2)

                elif method == "TSAA":
                    NMI_TSAA_complex_large[i,j] = calcNMI(file1,file2)
            
            i += 1

    df1 = pd.DataFrame(NMI_RBOAA_complex_large.T, columns = K_list)
    df2 = pd.DataFrame(NMI_OAA_complex_large.T, columns = K_list)
    df3 = pd.DataFrame(NMI_AA_complex_large.T, columns = K_list)
    df4 = pd.DataFrame(NMI_TSAA_complex_large.T, columns = K_list)

    df1['Method'] = 'RBOAA'
    df2['Method'] = 'OAA'
    df3['Method'] = 'AA'
    df4['Method'] = 'TSAA'


    df1 = df1.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    df2 = df2.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    df3 = df3.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    df4 = df4.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')


    df = pd.concat([df1,df2,df3,df4])
    df.Method = df.Method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'AA': 'AA', 'CAA': 'AA','TSAA':'TSAA'})

    methods = df['Method'].unique()
    fig, ax = plt.subplots(1,1,figsize = (15,5), layout='constrained')

    ax = sns.boxplot(x='Archetypes', y="NMI", hue="Method", showmeans=True, data=df,palette=methods_colors,meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    ax.set_xlabel('Number of archetypes', fontsize=30)
    ax.set_ylabel('NMI', fontsize=30)

    ax.set_ylim([0,1.05])
    plt.legend(fontsize=30,loc = 'upper right')

    if savedir is not None:
        plt.savefig(f'{savedir}/NMI_stability_plot.png', dpi=1000)
    

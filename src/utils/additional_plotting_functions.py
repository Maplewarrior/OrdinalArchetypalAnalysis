"""
A file containing plots intended to be incorporated in plots_class.py who are kept here for now.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import itertools
from src.utils.eval_measures import NMI, MCC

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# TODO: Make some nice error handling for K_list. Would be nice to have a default value for K_list
def loss_archetype_plot(K_list, results_path: str = 'synthetic_results/1000_complex_results.json'):
    """
    A plot over the final loss obtained as a function of the number of archetypes.
    Parameters:
        - results_path (str): Path to a .json results file created by running ResultMaker.get_results()
    """

    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)

    df_res.method = df_res.method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'AA': 'AA', 'CAA': 'AA'})    
    methods = df_res['method'].unique()

    methods_colors = dict(zip(methods.tolist(), ["#EF476F", "#FFD166", "#06D6A0", "#073B4C"]))
    plt.figure(figsize=(15,5))

    def add_curve(analysis_archetypes, losses, is_min: bool, method: str):
        if is_min:
            plt.plot(analysis_archetypes, losses,"-o",c=methods_colors[method],label=f'{method}')
        else:
            # plt.scatter(analysis_archetypes, losses)
            plt.plot(analysis_archetypes, losses, alpha=0.3, c=methods_colors[method])
    
    for method in methods:
        df_losses = df_res.loc[df_res['method'] == method][['n_archetypes', 'loss']]
        analysis_archetypes = K_list #df_losses['n_archetypes'].unique().tolist()
        
        all_losses = [df_losses.loc[df_losses['n_archetypes'] == e]['loss'].values for e in analysis_archetypes]
        analysis_archetypes = list(map(str, analysis_archetypes))
        losses = np.array([[e[-1] for e in loss] for loss in all_losses]) # n_archetypes x n_repeats array with losses at final iter
        
        tmp = losses[losses == np.min(losses, axis=1)[:, None]]
        _, idx = np.unique(losses[losses == np.min(losses, axis=1)[:, None]], return_index=True)
        min_losses = tmp[np.sort(idx)]
        
        add_curve(analysis_archetypes, min_losses, is_min=True, method=method)

        for rep in range(losses.shape[1]):
            add_curve(analysis_archetypes, losses[:, rep], is_min=False, method=method)

    
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)    
        plt.xlabel('Number of archetypes', fontsize=25)
        plt.ylabel('Loss', fontsize=25)
        plt.legend(fontsize=25)
    plt.show()

# loss_archetype_plot(df_res_20)

def NMI_archetypes(K_list, results_path: str = 'synthetic_results/1000_complex_results.json'):
    """
    A plot over the final loss obtained as a function of the number of archetypes
    """
    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)
    
    df_res = df_res[df_res['n_archetypes'].isin(K_list)]
    df_res.method = df_res.method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'AA': 'AA', 'CAA': 'AA'})

    
    methods = df_res['method'].unique()
    methods_colors = dict(zip(methods.tolist(), ["#EF476F", "#FFD166", "#06D6A0", "#073B4C"]))
    fig, ax = plt.subplots(1,1,figsize = (15,5), layout='constrained')

    ax = sns.boxplot(x='n_archetypes', y="NMI", hue="method", showmeans=True, data=df_res,palette=methods_colors,meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)    
    ax.set_xlabel('Number of archetypes', fontsize=25)
    ax.set_ylabel('NMI', fontsize=25)

    ax.set_ylim([0,1.05])
    plt.legend(fontsize=25,loc="upper right")

    plt.show()


K_list = [2,3,4,5,6,7,8,9,10]
#TODO : Lots of room for improvement with dataloader
def plot_NMI_stability(folder_path, K_list, repetitions= 10):
    methods = ["CAA", "OAA", "RBOAA"]


    test = itertools.combinations(range(repetitions), 2)
    t = list(test)
    calcIDX = np.array(t)

    NMI_RBOAA_complex_large = np.zeros((len(K_list),len(calcIDX)))
    NMI_OAA_complex_large = np.zeros((len(K_list),len(calcIDX)))
    NMI_AA_complex_large = np.zeros((len(K_list),len(calcIDX)))

    for method in methods:
        for K in K_list:
            for j in range(len(calcIDX)):
                
                filename1 = folder_path+"/A_"+str(method)+"_K="+str(K)+"_rep="+str(calcIDX[j,0])+".npy"
                filename2 = folder_path+"/A_"+str(method)+"_K="+str(K)+"_rep="+str(calcIDX[j,1])+".npy"
                file1 = np.load(filename1)
                file2 = np.load(filename1)

                if method == "RBOAA":
                    NMI_RBOAA_complex_large[K-2,j] = NMI(file1,file2)

                elif method == "OAA":
                    NMI_OAA_complex_large[K-2,j] = NMI(file1,file2)

                elif method == "CAA":
                    NMI_AA_complex_large[K-2,j] = NMI(file1,file2)

    df1 = pd.DataFrame(NMI_RBOAA_complex_large.T, columns = K_list)
    df2 = pd.DataFrame(NMI_OAA_complex_large.T, columns = K_list)
    df3 = pd.DataFrame(NMI_AA_complex_large.T, columns = K_list)

    df1['Method'] = 'RBOAA'
    df2['Method'] = 'OAA'
    df3['Method'] = 'AA'

    df1 = df1.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    df2 = df2.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    df3 = df3.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')


    df = pd.concat([df1,df2,df3])
    df.Method = df.Method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'AA': 'AA', 'CAA': 'AA'})

        
    methods = df['Method'].unique()
    methods_colors = dict(zip(methods.tolist(), ["#EF476F", "#FFD166", "#06D6A0", "#073B4C"]))
    fig, ax = plt.subplots(1,1,figsize = (15,5), layout='constrained')

    ax = sns.boxplot(x='Archetypes', y="NMI", hue="Method", showmeans=True, data=df,palette=methods_colors,meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)    
    ax.set_xlabel('Number of archetypes', fontsize=25)
    ax.set_ylabel('NMI', fontsize=25)

    ax.set_ylim([0,1.05])
    plt.legend(fontsize=25)
    plt.show()


            


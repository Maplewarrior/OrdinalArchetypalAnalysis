import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def NMI_archetypes(K_list, 
                   results_path: str,
                   results_path2: str,
                   methods_colors: dict, 
                   savedir: str = None):
    """
    A plot over NMI as a function of the number of archetypes.
    """
    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)

    # if results_path2:
    #     with open(f'{results_path2}', 'r') as f:
    #         result2 = json.load(f)
    #         df_res2 = pd.DataFrame(result2)

    #     df_res2.method = df_res2.method.replace({'CAA': 'TSAA'})    
    #     df_res = pd.concat([df_res,df_res2])
        
    df_res.method = df_res.method.replace({'RBOAA': 'RBOAA', 'OAA': 'OAA', 'CAA': 'AA','TSAA': 'TSAA'})  

    df_res = df_res[df_res['n_archetypes'].isin(K_list)]

    if "NMI" not in df_res.columns:
        return
    
    fig, ax = plt.subplots(1,1,figsize = (15,5), layout='constrained')

    ax = sns.boxplot(x='n_archetypes', y="NMI", hue="method", showmeans=True, data=df_res,palette=methods_colors,meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    ax.set_xlabel('Number of archetypes', fontsize=30)
    ax.set_ylabel('NMI', fontsize=30)

    ax.set_ylim([0,1.05])
    plt.legend(fontsize=30,loc="upper right")

    if savedir is not None:
        plt.savefig(f'{savedir}/NMI_archetype_plot.png', dpi=1000)
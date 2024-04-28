"""
A file containing plots intended to be incorporated in plots_class.py who are kept here for now.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from colormap import rgb2hex, rgb2hls, hls2rgb
import matplotlib.pyplot as plt

import itertools
from src.utils.eval_measures import NMI, MCC

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# TODO: Make some nice error handling for K_list. Would be nice to have a default value for K_list
def loss_archetype_plot(K_list, results_path: str = 'synthetic_results/1000_complex_results.json',results_path2: str = 'C:/Users/aejew/Downloads/AA_results/AA_results/naive_OSM_results/all_AA_results.json'):
    """
    A plot over the final loss obtained as a function of the number of archetypes.
    Parameters:
        - results_path (str): Path to a .json results file created by running ResultMaker.get_results()
    """

    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)


    with open(f'{results_path2}', 'r') as f:
        result2 = json.load(f)
        df_res2 = pd.DataFrame(result2)

    df_res2.method = df_res2.method.replace({'CAA': 'TSAA'})    
    df_res.method = df_res.method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'CAA': 'AA','TSAA': 'TSAA'})  

    df_res = pd.concat([df_res,df_res2])  
    methods = df_res['method'].unique()

    methods_colors = dict(zip(methods.tolist(), ["#EF476F", "#FFD166", "#06D6A0", "#073B4C"]))
    fig, ax = plt.subplots(figsize = (15,5), layout='constrained')

    def add_curve(analysis_archetypes, losses, is_min: bool, method: str):
        if is_min:
            plt.plot(analysis_archetypes, losses,"-o",c=methods_colors[method],label=f'{method}')
        else:
            # plt.scatter(analysis_archetypes, losses)
            plt.plot(analysis_archetypes, losses, alpha=0.3, c=methods_colors[method])

    ax.set_xlabel('Number of archetypes', fontsize=25)
    ax.set_ylabel('Cross entropy loss', fontsize=25)
    ax2 = ax.twinx()
    ax2.set_ylabel('SSE',fontsize=25)

    # TODO: Automatic set of ax2 ylim to avoid the two plots starting the same place something likemin = min(min_loss_AA, min_loss_TSOAA)*0.9 and max  = max(max_loss_AA, max_loss_TSOAA)*1.1
    #ax2.set_ylim((np.min(df_res[df_res.method=='AA'].loss[-1])-1000,np.max(df_res[df_res.method=='TSAA'].loss[0])+1000))
    #ax.set_ylim((np.min(df_res[df_res.method=='RBOAA'].loss[-1])-1000,np.max(df_res[df_res.method=='OAA'].loss[0])+1000))
    


    for method in methods:
        df_losses = df_res.loc[df_res['method'] == method][['n_archetypes', 'loss']]
        analysis_archetypes = K_list #df_losses['n_archetypes'].unique().tolist()

        all_losses = [df_losses.loc[df_losses['n_archetypes'] == e]['loss'].values for e in analysis_archetypes]
        analysis_archetypes = list(map(str, analysis_archetypes))
        losses = np.array([[e[-1] for e in loss] for loss in all_losses]) # n_archetypes x n_repeats array with losses at final iter
        
        tmp = losses[losses == np.min(losses, axis=1)[:, None]]
        _, idx = np.unique(losses[losses == np.min(losses, axis=1)[:, None]], return_index=True)
        min_losses = tmp[np.sort(idx)]

        
        if method in ['AA','TSAA']:
            
            ax2.plot(analysis_archetypes, min_losses,"-o",c=methods_colors[method],label=f'{method}')
            # = add_curve(analysis_archetypes, min_losses, is_min=True, method=method)

            for rep in range(losses.shape[1]):
                ax2.plot(analysis_archetypes, losses[:, rep], alpha=0.3, c=methods_colors[method])
        
        else:
            ax.plot(analysis_archetypes, min_losses,"-o",c=methods_colors[method],label=f'{method}')
            #ax.add_curve(analysis_archetypes, min_losses, is_min=True, method=method)

            for rep in range(losses.shape[1]):
                ax.plot(analysis_archetypes, losses[:, rep], alpha=0.3, c=methods_colors[method])
    

    # Not sure what diff between minor and major is

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax2.tick_params(axis='both', which='major', labelsize=20,colors = "#073B4C")
    ax2.tick_params(axis='both', which='minor', labelsize=20)

    #ax2.set_ylim[2000,11000]

    ax2.spines['right'].set_color(methods_colors["AA"])
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0,fontsize=25)


# loss_archetype_plot(df_res_20)

def NMI_archetypes(K_list, results_path: str = 'synthetic_results/1000_complex_results.json',results_path2: str = 'C:/Users/aejew/Downloads/AA_results/AA_results/naive_OSM_results/all_AA_results.json'):
    """
    A plot over the final loss obtained as a function of the number of archetypes
    """
    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)

    
    with open(f'{results_path2}', 'r') as f:
        result2 = json.load(f)
        df_res2 = pd.DataFrame(result2)

    df_res2.method = df_res2.method.replace({'CAA': 'TSAA'})    
    df_res.method = df_res.method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'CAA': 'AA','TSAA': 'TSAA'})  

    df_res = pd.concat([df_res,df_res2])      
    df_res = df_res[df_res['n_archetypes'].isin(K_list)]

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



def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))

def darken_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 - factor)


def plot_archetypal_answers(X,archetypes,likert_text,questions,startColor, type = 'points'):

    def transform_data(data, p):
        likert_counts = pd.DataFrame(columns = range(1,p+1), index = np.arange(data.shape[0]))

        for i in range(data.shape[0]):
            likert_counts.iloc[i,(np.unique(data[i,:], return_counts=True)[0]-1)] = np.unique(data[i,:], return_counts=True)[1]

        likert_counts = likert_counts.fillna(0)

        return likert_counts
    
    likert_counts = transform_data(X, 5)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(likert_counts.values,aspect='auto', cmap = 'Greys', alpha = 0.8)
    cbar = ax.figure.colorbar(ax.imshow(likert_counts.values,aspect='auto', cmap = 'Greys', alpha = 0.8))

    ax.set_xticks(np.arange(0,5))
    ax.set_xticklabels(likert_text, rotation = 45)

    ax.set_yticks(np.arange(0,likert_counts.shape[0]))
    ax.set_yticklabels(questions)

    y = np.arange(likert_counts.shape[0])

    
    color = []
    color += [startColor]

    ## make off set such that middle archetype is centered
    center = (archetypes.shape[1])//2
    offset = (np.arange(archetypes.shape[1])-center)*0.1
    



    for i in range(archetypes.shape[1]):
        r, g, b  = hex_to_rgb(color[i])
        color += [darken_color(r, g, b,0.5)]

        if type == 'points':
            ax.scatter(archetypes[:,i]-1+offset[i], y, lw=5., color=color[i],label = f'Archetype {i+1}')




        else:
            line = plt.Line2D(archetypes[:,i]-1, y, lw=5., color=color[i],label = f'Archetype {i+1}')
            line.set_clip_on(False)
            ax.add_line(line)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
         


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

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
### GLOBAL PARAMS
my_pallette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSOAA" : "#073B4C", "GT": "#7E99DC"}
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = True # allow for latex axes




# loss_archetype_plot(df_res_25)




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
         

### Helper function for Response Bias Plot:
def get_alphas_from_betas(X, RBOAA_betas, OAA_betas, synthetic_betas):
    alpha_OAA = []
    for j in range(len(OAA_betas)-1):
        alpha_OAA += [(OAA_betas[j+1]+OAA_betas[j])/2]
    
    ### convert [0, 1]
    alpha_OAA = np.array(alpha_OAA)
    neg_mask = alpha_OAA < 0
    one_mask = alpha_OAA > 1
    alpha_OAA[neg_mask] = 0
    alpha_OAA[one_mask] = 1
    alpha_OAA = list(alpha_OAA)
    
    alpha = np.zeros([X.shape[1], len(OAA_betas) - 1])

    for i in range(X.shape[1]):
        for j in range(RBOAA_betas.shape[1]-1):
            alpha_val = (RBOAA_betas[i,j+1]+RBOAA_betas[i,j])/2
            ### constrain to [0, 1]
            alpha_val = 1 if alpha_val > 1 else alpha_val
            alpha_val = 0 if alpha_val < 0 else alpha_val
            alpha[i,j] = alpha_val
    
    if synthetic_betas is not None:
        if synthetic_betas.ndim > 1:
            synthetic_alphas = np.empty((synthetic_betas.shape[0], synthetic_betas.shape[1]-1))

            for i in range(synthetic_betas.shape[0]):
                for j in range(synthetic_betas.shape[1]-1):
                    synthetic_alphas[i, j] = (synthetic_betas[i,j] + synthetic_betas[i, j+1]) / 2
        else:
            synthetic_alphas = np.empty(synthetic_betas.shape[0]-1)
            for j in range(synthetic_betas.shape[0] - 1):
                synthetic_alphas[j] = (synthetic_betas[j] + synthetic_betas[j+1]) / 2
    else:
        synthetic_alphas = None
                
    return alpha_OAA, alpha, synthetic_alphas 

def response_bias_plot(X, RBOAA_betas, OAA_betas, synthetic_betas=None):
    """
    X: M x N sized array
    RBOAA_betas: N x p+1 array
    OAA_betas: p+1 array
    synthetic betas p+1 or N x p+1 array depending on if RB = False/True
    """
    r, g, b  = hex_to_rgb(my_pallette['OAA'])
    OAA_color = [darken_color(r, g, b,0.5)]

    alpha_OAA, alpha, synthetic_alphas = get_alphas_from_betas(X, RBOAA_betas, OAA_betas, synthetic_betas)

    fig, ax = plt.subplots(1,1, figsize = (15,5), layout='constrained')
    
    ### plot RBOAA betas
    if (synthetic_betas is not None) and (synthetic_betas.ndim > 1): # Synthetic analysis
        if synthetic_betas.ndim > 1:
            
            method_colors = {'RBOAA': my_pallette['RBOAA'], 'Ground truth': my_pallette['GT']}
            df1 = pd.DataFrame(synthetic_alphas, columns=[f'{i}' for i in range(1,6)])
            df2 = pd.DataFrame(alpha, columns=[f'{i}' for i in range(1,6)])

            df1.loc[:, 'Method'] = 'Ground truth'
            df2.loc[:, 'Method'] = 'RBOAA'

            df1 = df1.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df2 = df2.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df = pd.concat([df1, df2])

            sns.boxplot(x='Point', y="alpha", hue="Method", showmeans=False, data=df, 
                        palette=method_colors, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
            
    else: # Real data
        medianprops = dict(linewidth=2.5, color=my_pallette['RBOAA'], size=500)
        rboaa_res = ax.boxplot(alpha, medianprops=medianprops)
        # plot ground truth
        if synthetic_betas is not None:
            gt_res = ax.scatter(x= ax.get_xticks(), y=synthetic_alphas, marker='X',s=170, color=my_pallette['GT'], label = 'Ground truth')
    
    oaa_res = ax.scatter(x= ax.get_xticks(), y=alpha_OAA,marker='X',s=170, color = OAA_color,label = 'OAA', zorder=10)
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
        
    # print(df.Answer)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    ax.set_xlabel('Point on likert scale', fontsize=30)
    ax.set_ylabel(r"$\alpha$", fontsize=30)

    # likert_text = ['1. very much like me', '2. like me', '3. somewhat like me', '4. A little like me', '5. Not like me', '6. Not like me at all']
    
    # ax.set_xticklabels(likert_text, rotation = 15)
    ax.set_ylim([0,1.05])

    
    dummy_boxplot_rboaa = plt.Line2D([0], [0], linestyle='-', color=my_pallette['RBOAA'], linewidth=2.5)
    dummy_boxplot_gt = plt.Line2D([0], [0], linestyle='-', color=my_pallette['GT'], linewidth=2.5)
    
    if synthetic_betas is not None:
        ax.legend([oaa_res, dummy_boxplot_rboaa, dummy_boxplot_gt], ['OAA', 'RBOAA', 'Ground truth'], fontsize=30, loc='upper left')
    else:
        ax.legend([oaa_res, dummy_boxplot_rboaa], ['OAA', 'RBOAA'], fontsize=30, loc='upper left')

    plt.savefig("Plots_for_paper/RB_Complex.png",dpi=1000)



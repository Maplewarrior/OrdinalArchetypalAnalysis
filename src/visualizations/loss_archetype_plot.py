import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
### GLOBAL PARAMS
my_pallette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSOAA" : "#073B4C", "GT": "#7E99DC"}
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.rcParams['text.usetex'] = True # allow for latex axes

# TODO: Make some nice error handling for K_list. Would be nice to have a default value for K_list
def loss_archetype_plot(K_list, results_path: str = 'synthetic_results/1000_complex_results.json',results_path2: str = None):
    """
    A plot over the final loss obtained as a function of the number of archetypes.
    Parameters:
        - results_path (str): Path to a .json results file created by running ResultMaker.get_results()
    """

    my_pallette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSOAA" : "#073B4C", "GT": "#7E99DC"}
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)


    if results_path2:
        with open(f'{results_path2}', 'r') as f:
            result2 = json.load(f)
            df_res2 = pd.DataFrame(result2)

        df_res2.method = df_res2.method.replace({'CAA': 'TSAA'})    
        df_res = pd.concat([df_res,df_res2])

    df_res.method = df_res.method.replace({'OAA': 'OAA', 'RBOAA': 'RBOAA', 'CAA': 'AA','TSAA': 'TSAA'})  

    methods = df_res['method'].unique()

    methods_colors = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSAA" : "#073B4C"}
    #methods_colors = dict(zip(methods.tolist(), ["#EF476F", "#FFD166", "#06D6A0", "#073B4C"]))
    fig, ax = plt.subplots(figsize = (15,5), layout='constrained')

    def add_curve(analysis_archetypes, losses, is_min: bool, method: str):
        if is_min:
            plt.plot(analysis_archetypes, losses,"-o",c=methods_colors[method],label=f'{method}')
        else:
            # plt.scatter(analysis_archetypes, losses)
            plt.plot(analysis_archetypes, losses, alpha=0.3, c=methods_colors[method])

    ax.set_xlabel('Number of archetypes', fontsize=30)
    ax.set_ylabel('Cross entropy loss', fontsize=30)
    ax2 = ax.twinx()
    ax2.set_ylabel('SSE',fontsize=30)

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

        print(str(method)+str(idx))
        print(tmp)

        
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
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    ax2.tick_params(axis='both', which='major', labelsize=25,colors = "#073B4C")
    ax2.tick_params(axis='both', which='minor', labelsize=25)

    #ax2.set_ylim[2500,11000]

    ax2.spines['right'].set_color(methods_colors["AA"])
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0,fontsize=30)

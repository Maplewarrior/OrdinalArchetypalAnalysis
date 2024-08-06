
from src.visualizations.functions import get_alphas_from_betas, hex_to_rgb, darken_color
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd


def response_bias_plot(X, RBOAA_betas, OAA_betas, p, savedir: str, my_pallette: dict, synthetic_betas=None):
    """
    X: M x N sized array
    RBOAA_betas: N x p+1 array
    OAA_betas: p+1 array
    synthetic betas p+1 or N x p+1 array depending on if RB = False/True
    """

    # matplotlib.rcParams['mathtext.fontset'] = 'stix'
    # matplotlib.rcParams['font.family'] = 'STIXGeneral'
    r, g, b  = hex_to_rgb(my_pallette['OAA'])
    OAA_color = darken_color(r, g, b,0.5)

    r, g, b  = hex_to_rgb(my_pallette['RBOAA'])
    RBOAA_color = darken_color(r, g, b,0.5)

    alphaOAA, alphaRBOAA, synthetic_alphas = get_alphas_from_betas(X, RBOAA_betas, OAA_betas, synthetic_betas)
    # alpha_OAAQ100, alphaQ100, synthetic_alphas = get_alphas_from_betas(X, RBOAA_betasQ100, OAA_betasQ100, synthetic_betas)

    fig, ax = plt.subplots(1,1, figsize = (5,8), layout='constrained')
    
    ### plot RBOAA betas
    if (synthetic_betas is not None) and (synthetic_betas.ndim > 1): # Synthetic analysis
        my_pallette.update({'Ground truth': 'b'})
        if synthetic_betas.ndim > 1:
            
            df1 = pd.DataFrame(synthetic_alphas, columns=[f'{i}' for i in range(1,p+1)])
            df2 = pd.DataFrame(alphaRBOAA, columns=[f'{i}' for i in range(1,p+1)])
            
            df1.loc[:, 'Method'] = 'Ground truth'
            df2.loc[:, 'Method'] = 'RBOAA'

            df1 = df1.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df2 = df2.melt(id_vars='Method', var_name='Point', value_name='alpha')
            
            df = pd.concat([df1, df2])

            sns.boxplot(y='Point', x="alpha", hue="Method", showmeans=False, data=df, 
                        palette=my_pallette,vert=False, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
            
    else: # Real data
        medianprops = dict(linewidth=2.5, color=my_pallette['RBOAA'])
        df1 = pd.DataFrame(alphaRBOAA, columns=[f'{i}' for i in range(1,p+1)])
        
        df1.loc[:,'Method'] ='RBOAA'
        df = df1.melt(id_vars='Method', var_name='Point', value_name='alpha')
        sns.boxplot(y='Point', x="alpha", hue="Method", showmeans=False, data=df, 
                        palette=my_pallette,vert=False, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    
        # plot ground truth
        if synthetic_betas is not None:
            gt_res = ax.scatter(y= ax.get_yticks(), x=synthetic_alphas, marker='X',s=170, color='b', label = 'Ground truth')
    

    oaa_res = ax.scatter(y= ax.get_yticks(), x=alphaOAA,marker='o',s=300, color = OAA_color,label = 'OAA', zorder=10)
    ax.yaxis.grid(True, which='major')
    [ax.axhline(x+.5,color='k') for x in ax.get_yticks()]
        
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    ax.set_ylabel('Point on Likert scale', fontsize=30)
    ax.set_xlabel(r'$\alpha$', fontsize=30)

    #ax.set_xticklabels(likert_text, rotation = 15)
    ax.set_xlim([-0.01,1.05])

    
    # dummy_boxplot_rboaaQ20 = plt.Line2D([0], [0], linestyle='-', color=my_pallette['RBOAA'], linewidth=2.5)
    # dummy_boxplot_rboaaQ100 = plt.Line2D([0], [0], linestyle='-', color='r', linewidth=2.5)
    # dummy_boxplot_gt = plt.Line2D([0], [0], linestyle='-', color='b', linewidth=2.5)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2, fancybox=True, shadow=True,fontsize=15)

    plt.savefig(f"{savedir}/response_bias_plot.png",dpi=1000)


def response_bias_plot_multiple(X, RBOAA_betasQ20, OAA_betasQ20,RBOAA_betasQ100, OAA_betasQ100, p, plotname, synthetic_betas=None):
    """
    X: M x N sized array
    RBOAA_betas: N x p+1 array
    OAA_betas: p+1 array
    synthetic betas p+1 or N x p+1 array depending on if RB = False/True
    """

    my_pallette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSOAA" : "#073B4C","GT": "#9055A2"}
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    r, g, b  = hex_to_rgb(my_pallette['OAA'])
    OAA_color = darken_color(r, g, b,0.5)

    r, g, b  = hex_to_rgb(OAA_color)
    OAA_color2 = darken_color(r, g, b,0.5)

    r, g, b  = hex_to_rgb(my_pallette['RBOAA'])
    RBOAA_color = darken_color(r, g, b,0.5)

    alpha_OAAQ20, alphaQ20, synthetic_alphas = get_alphas_from_betas(X, RBOAA_betasQ20, OAA_betasQ20, synthetic_betas)
    alpha_OAAQ100, alphaQ100, synthetic_alphas = get_alphas_from_betas(X, RBOAA_betasQ100, OAA_betasQ100, synthetic_betas)

    fig, ax = plt.subplots(1,1, figsize = (5,8), layout='constrained')
    method_colors = {'RBOAA Q20': my_pallette['RBOAA'], 'RBOAA Q100': RBOAA_color ,'Ground truth':'b'}
    
    ### plot RBOAA betas
    if (synthetic_betas is not None) and (synthetic_betas.ndim > 1): # Synthetic analysis
        if synthetic_betas.ndim > 1:
            
            df1 = pd.DataFrame(synthetic_alphas, columns=[f'{i}' for i in range(1,p)])
            df2 = pd.DataFrame(alphaQ100, columns=[f'{i}' for i in range(1,p)])
            df3 = pd.DataFrame(alphaQ20, columns=[f'{i}' for i in range(1,p)])

            df1.loc[:, 'Method'] = 'Ground truth'
            df2.loc[:, 'Method'] = 'RBOAA Q100'
            df3.loc[:, 'Method'] = 'RBOAA Q20'

            df1 = df1.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df2 = df2.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df3 = df3.melt(id_vars='Method', var_name='Point', value_name='alpha')
            df = pd.concat([df1, df2, df3])

            sns.boxplot(y='Point', x="alpha", hue="Method", showmeans=False, data=df, 
                        palette=method_colors,vert=False, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
            
    else: # Real data
        medianprops = dict(linewidth=2.5, color=my_pallette['RBOAA'])
        df1 = pd.DataFrame(alphaQ20, columns=[f'{i}' for i in range(1,p)])
        df2 = pd.DataFrame(alphaQ100, columns=[f'{i}' for i in range(1,p)])
        
        df1.loc[:,'Method'] ='RBOAA Q20'
        df2.loc[:,'Method'] = 'RBOAA Q100'
        df1 = df1.melt(id_vars='Method', var_name='Point', value_name='alpha')
        df2 = df2.melt(id_vars='Method', var_name='Point', value_name='alpha')

        df = pd.concat([df1, df2])

        sns.boxplot(y='Point', x="alpha", hue="Method", showmeans=False, data=df, 
                        palette=method_colors,vert=False, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    
        # plot ground truth
        if synthetic_betas is not None:
            gt_res = ax.scatter(y= ax.get_yticks(), x=synthetic_alphas, marker='X',s=170, color='b', label = 'Ground truth')
    

    oaa_res = ax.scatter(y= ax.get_yticks(), x=alpha_OAAQ20,marker='o',s=300, color = OAA_color,label = 'OAA Q20', zorder=10)
    oaa_res2 = ax.scatter(y= ax.get_yticks(), x=alpha_OAAQ100,marker='X',s=170, color = OAA_color2,label = 'OAA Q100', zorder=10)
    ax.yaxis.grid(True, which='major')
    [ax.axhline(x+.5,color='k') for x in ax.get_yticks()]
        
    # print(df.Answer)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    ax.set_ylabel('Point on Likert scale', fontsize=30)
    ax.set_xlabel(r'$\alpha$', fontsize=30)
    #ax.set_ylabel(r"$\alpha$", fontsize=30)

    # likert_text = ['1. very much like me', '2. like me', '3. somewhat like me', '4. A little like me', '5. Not like me', '6. Not like me at all']
    
    #ax.set_xticklabels(likert_text, rotation = 15)
    ax.set_xlim([-0.01,1.05])

    
    dummy_boxplot_rboaaQ20 = plt.Line2D([0], [0], linestyle='-', color=my_pallette['RBOAA'], linewidth=2.5)
    dummy_boxplot_rboaaQ100 = plt.Line2D([0], [0], linestyle='-', color='r', linewidth=2.5)
    dummy_boxplot_gt = plt.Line2D([0], [0], linestyle='-', color='b', linewidth=2.5)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2, fancybox=True, shadow=True,fontsize=15)

    plt.savefig(f"figures/{plotname}.png",dpi=1000)


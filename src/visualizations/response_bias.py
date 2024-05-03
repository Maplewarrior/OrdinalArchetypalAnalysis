


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


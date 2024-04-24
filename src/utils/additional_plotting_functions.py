"""
A file containing plots intended to be incorporated in plots_class.py who are kept here for now.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def loss_archetype_plot(results_path: str = 'synthetic_results/1000_complex_results.json'):
    """
    A plot over the final loss obtained as a function of the number of archetypes
    """
    with open(f'{results_path}', 'r') as f:
        result = json.load(f)
        df_res = pd.DataFrame(result)

    
    methods = df_res['method'].unique()
    methods_colors = dict(zip(methods.tolist(), ['r', 'g', 'b', 'y']))
    plt.figure(figsize=(6,4))

    def add_curve(analysis_archetypes, losses, is_min: bool, method: str):
        if is_min:
            plt.scatter(analysis_archetypes, losses, label=f'{method}', c=methods_colors[method])
            plt.plot(analysis_archetypes, losses, c=methods_colors[method])
        else:
            # plt.scatter(analysis_archetypes, losses)
            plt.plot(analysis_archetypes, losses, alpha=0.3, c=methods_colors[method])
    
    for method in methods:
        df_losses = df_res.loc[df_res['method'] == method][['n_archetypes', 'loss']]
        analysis_archetypes = df_losses['n_archetypes'].unique().tolist()
        
        all_losses = [df_losses.loc[df_losses['n_archetypes'] == e]['loss'].values for e in analysis_archetypes]
        analysis_archetypes = list(map(str, analysis_archetypes))
        losses = np.array([[e[-1] for e in loss] for loss in all_losses]) # n_archetypes x n_repeats array with losses at final iter
        
        tmp = losses[losses == np.min(losses, axis=1)[:, None]]
        _, idx = np.unique(losses[losses == np.min(losses, axis=1)[:, None]], return_index=True)
        min_losses = tmp[np.sort(idx)]
        
        add_curve(analysis_archetypes, min_losses, is_min=True, method=method)

        for rep in range(losses.shape[1]):
            add_curve(analysis_archetypes, losses[:, rep], is_min=False, method=method)
        
        plt.xlabel('n.o. archetypes')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f'Final loss as a function of number of archetypes.')
    plt.show()
    # print(losses)

# loss_archetype_plot(df_res_20)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualizations.functions import hex_to_rgb, darken_color, isNan

def plot_archetypal_answers(X,archetypes, p: int, likert_text: list[str], questions: list[str], startColor, type = 'points', savepath: str = None):

    def transform_data(data, p):
        likert_counts = pd.DataFrame(columns = range(1,p+1), index = np.arange(data.shape[0]))

        for i in range(data.shape[0]):
            likert_counts.iloc[i,(np.unique(data[i,:], return_counts=True)[0]-1)] = np.unique(data[i,:], return_counts=True)[1]

        # replace nan values with 0
        nan_mask = isNan(likert_counts.value_counts)
        likert_counts[nan_mask] = 0

        return likert_counts
    
    likert_counts = transform_data(X, p)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(likert_counts.values,aspect='auto', cmap = 'Greys', alpha = 0.8)

    ax.set_xticks(np.arange(0,p))
    ax.set_xticklabels(likert_text, rotation = 45)
    ax.set_yticks(np.arange(0,likert_counts.shape[0]))
    ax.set_yticklabels(questions)

    y = np.arange(likert_counts.shape[0])
    color = [startColor]

    ## make off set such that middle archetype is centered
    center = (archetypes.shape[1])//2
    offset = (np.arange(archetypes.shape[1])-center)*0.1

    for i in range(archetypes.shape[1]):
        r, g, b  = hex_to_rgb(color[i])
        color += [darken_color(r, g, b,0.5)]

        if type == 'points':
            ax.scatter(archetypes[:,i]-1+offset[i], y, lw=p, color=color[i],label = f'Archetype {i+1}')

        else:
            ax.plot(archetypes[:,i]-1+offset[i], y,'-o' ,lw=2., color=color[i],label = f'Archetype {i+1}')
            # line = plt.Line2D(archetypes[:,i]-1, y, lw=2., color=color[i],label = f'Archetype {i+1}')
            # line.set_clip_on(False)
            # ax.add_line(line)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    
    if savepath is not None:
        plt.savefig(savepath, dpi=1000)


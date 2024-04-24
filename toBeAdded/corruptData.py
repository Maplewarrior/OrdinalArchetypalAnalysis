import numpy as np

def corruptData(data, corruption_rate,likertScale):
    np.random.seed(0)

    ## Number of data points to corrupt
    cor_Perc = data.size*corruption_rate

    # Find index
    idx_cor1 = np.random.choice(data.shape[1], int(cor_Perc), replace=False)
    idx_cor2 = np.random.uniform(0, data.shape[0], int(cor_Perc)).astype(int)

    data[idx_cor2, idx_cor1] = np.random.uniform(1, likertScale, int(cor_Perc))

    


    return data, idx_cor1, idx_cor2
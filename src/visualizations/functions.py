from colormap import rgb2hex, rgb2hls, hls2rgb
import numpy as np

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
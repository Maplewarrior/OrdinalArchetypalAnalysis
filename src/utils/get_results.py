import pdb
from src.utils.result_helper_function import result_helper_function
import multiprocessing

""" Define parameters:
    - a_param:   Affects the weighting of archetypes on respondents
    - b_param:   Affects the response bias in the synthetic data, low value --> high RB
    - sigma:     Noise parameter, high value --> high noise
    - sigma_dev: Determines the variation in sigma when modelled on each individual"""

def get_all_results():

    # define parameters
    a_params = [0.85, 1, 2]
    b_params = [1, 5, 10]
    sigmas = [-3.5]#[]
    sigma_stds = [0]

    n_synthetic_archetypes = 5
    varying_analysis_archetypes = False
    all_parameters = []

    # Loop over parameters
    for sigma in sigmas:
        for sigma_std in sigma_stds:
            for a_param in a_params:
                for b_param in b_params:
                    # set parameters to pass to results_helper_function
                    all_parameters.append([sigma, n_synthetic_archetypes, a_param, b_param, varying_analysis_archetypes, sigma_std])
                    params = [sigma, n_synthetic_archetypes, a_param, b_param, varying_analysis_archetypes, sigma_std]
                    result_helper_function(params)
    #with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
#        p.map(result_helper_function, all_parameters)

if __name__ == '__main__':
    get_all_results()
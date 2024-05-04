import json
import numpy as np
import pdb
import pandas as pd
from src.inference.ResultMaker import ResultMaker


def mean_across_runs(filepath: str = 'results/test_run1.json'):
    with open(filepath, "r") as f:
        res = json.load(f)
        df_res = pd.DataFrame.from_dict(res)

    methods = ['CAA', 'OAA', 'RBOAA']
    bool_list = [True, False]

    # for method in methods:
    #     for _init in bool_list:
    #         for alternating in bool_list:
    #             for beta_reg in bool_list:
    means = df_res[['method', 'with_init', 'alternating', 'beta_reg', 'n_archetypes', 'NMI', 'MCC']].groupby(by=['method', 'with_init', 'alternating', 'beta_reg', 'n_archetypes']).mean()
    print(means)
    return df_res.loc[(df_res['alternating'] == False) & (df_res['method'] == 'RBOAA') & (df_res['with_init'] == True) & (df_res['beta_reg'] == True)]

    # means = df_res.groupby(by=['method', 'with_init', 'beta_reg', 'alternating']).mean()

def main():

    #### COMPLEX PARAMETERS
    complex_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':20, # irrelevant if load_data = True
                            'K':3, 
                            'p':5, 
                            'rb': True,
                            'mute': True,
                            'savefolder': 'complex_results',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }
    
    complex_100Q_data_params = {'N':1000, # irrelevant if load_data = True
                                'M':100, # irrelevant if load_data = True
                                'K':3,
                                'p':5, 
                                'rb': True,
                                'mute': True,
                                'savefolder': 'complex_100Q_results',
                                'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                                'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                                'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                                'A_path': None #'SyntheticData/1000_respondents/A.npy'
                                }
    
    no_RB_100Q_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':100, # irrelevant if load_data = True
                            'K':3,
                            'p':5,
                            'rb': False,
                            'mute': True,
                            'savefolder': 'noRB_100Q_results',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }
    no_RB_20Q_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':20, # irrelevant if load_data = True
                            'K':3,
                            'p':5,
                            'rb': False,
                            'mute': True,
                            'savefolder': 'noRB_20Q',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }
    
    RB_100Q_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':100, # irrelevant if load_data = True
                            'K':3,
                            'p':5,
                            'rb': True,
                            'mute': True,
                            'savefolder': 'RB_100Q',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }
    
    RB_20Q_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':20, # irrelevant if load_data = True
                            'K':3,
                            'p':5,
                            'rb': True,
                            'mute': True,
                            'savefolder': 'RB_20Q',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }

    complex_OSM_data_params = {
        'N':1000, # irrelevant if load_data = True
        'M':20, # irrelevant if load_data = True
        'K':3, 
        'p':5, 
        'rb': True,
        'mute': True,
        'savefolder': 'complex_OSM_results',
        'load_data': True,
        'X_path': 'SyntheticData/1000_respondents_complex/OSM/data_complex_OSM_large.csv',
        'Z_path': 'SyntheticData/1000_respondents_complex/Z.npy',
        'A_path': 'SyntheticData/1000_respondents_complex/A.npy'}

    complex_corrupted_data_params = {'N':1000, # irrelevant if load_data = True
                                     'M':20, # irrelevant if load_data = True
                                     'K':3, 
                                     'p':5, 
                                     'rb': True,
                                     'mute': True,
                                     'savefolder': 'complex_corrupted_results',
                                     'load_data': True,
                                     'X_path': 'SyntheticData/1000_respondents_complex/Data_complex_large_corrupted.npz',
                                     'Z_path': 'SyntheticData/1000_respondents_complex/Z.npy',
                                     'A_path': 'SyntheticData/1000_respondents_complex/A.npy'}

    complex_corrupted_OSM_data_params = {'N':1000, # irrelevant if load_data = True
                                        'M':20, # irrelevant if load_data = True
                                        'K':3, 
                                        'p':5, 
                                        'rb': True,
                                        'mute': True,
                                        'savefolder': 'complex_OSM_corrupted_results',
                                        'load_data': True,
                                        'X_path': 'SyntheticData/1000_respondents_complex/OSM/data_complex_OSM_large_corrupted.csv',
                                        'Z_path': 'SyntheticData/1000_respondents_complex/Z.npy',
                                        'A_path': 'SyntheticData/1000_respondents_complex/A.npy'}
    
    #### ESS8 PARAMETERS
    ESS8_data_params = {'p': 6,
                        'mute': True,
                        'savefolder': 'ESS8_results',
                        'load_data': True,
                        'X_path': 'ESS8',
                        'Z_path': None,
                        'A_path': None}

    ESS8_GB_data_params = {'p': 6,
                           'mute': True,
                           'savefolder': 'ESS8_GB_results',
                           'load_data': True,
                           'X_path': 'ESS8_GB',
                           'Z_path': None,
                           'A_path': None}
    ESS8_GB_corrupted_data_params = {'p': 6,
                           'mute': True,
                           'savefolder': 'ESS8_GB_corrupted_results',
                           'load_data': True,
                           'X_path': 'RealData/GB_data_corrupted.npz',
                           'Z_path': None,
                           'A_path': None}
    
    ##### NO RB parameters
    no_RB_OSM_params = {'N':1000, # irrelevant if load_data = True
                        'M':20, # irrelevant if load_data = True
                        'K':3, 
                        'p':5, 
                        'rb': True,
                        'mute': True,
                        'savefolder': 'no_RB_OSM',
                        'load_data': True,
                        'X_path': 'SyntheticData/1000_respondents_noRB/OSM/no_RB_OSM.csv',
                        'Z_path': 'SyntheticData/1000_respondents_noRB/Z.npy',
                        'A_path': 'SyntheticData/1000_respondents_noRB/A.npy'}
    
    no_RB_OSM_corrupted_params = {'N':1000, # irrelevant if load_data = True
                        'M':20, # irrelevant if load_data = True
                        'K':3, 
                        'p':5, 
                        'rb': True,
                        'mute': True,
                        'savefolder': 'no_RB_OSM_corrupted',
                        'load_data': True,
                        'X_path': 'SyntheticData/1000_respondents_noRB/OSM/no_RB_OSM_corrupted.csv',
                        'Z_path': 'SyntheticData/1000_respondents_noRB/Z.npy',
                        'A_path': 'SyntheticData/1000_respondents_noRB/A.npy'}

    #### MODEL OPTIONS
    model_options = {'method': ['OAA', 'RBOAA'],
                    'with_init': [True],
                    'alternating': [False],
                    'beta_reg': [True],
                    'n_archetypes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                     20, 30, 40, 50]}
    model_options_OSM = {'method': [''],
                        'with_init': [True],
                        'alternating': [False],
                        'beta_reg': [True],
                        'n_archetypes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        20, 30, 40, 50]}
    model_options_test = {'method': ['OAA', 'RBOAA'],
                        'with_init': [True],
                        'alternating': [False],
                        'beta_reg': [True],
                        'n_archetypes': [1, 2, 3, 4, 5]}
    

    lrs = {} # hard coded for now
    RM = ResultMaker(data_params=no_RB_20Q_data_params, model_options=model_options, lrs=lrs)
    RM.get_results()
    # RM.get_ESS8_results()
    # ALL_DATA_PARAMS = [complex_data_params, complex_corrupted_data_params, naive_data_params, naive_corrupted_data_params]
    # for d_params in ALL_DATA_PARAMS:
    #     print("\n\nRunning analysis with: ", d_params['savefolder'],"\n\n")
    #     RM = ResultMaker(data_params=d_params, model_options=model_options, lrs=lrs)
    #     RM.get_results()

if __name__ == '__main__':
    main()

    # from src.utils.synthetic_data_class import _synthetic_data
    # SD = _synthetic_data(1000, 20, 3, 5, -9.21, True, 1., 1.5, 1e-6)
    # betas = SD.betas

    
    #### Make AA_results.json object
    # from src.inference.utils import make_aa_results_json, load_analyses
    # ESS8_gb_analyses = load_analyses('ESS8_GB_results_2')
    # make_aa_results_json(ESS8_gb_analyses, 'synthetic_results/ESS8_GB_results_2/All_AA_results.json')
    # pdb.set_trace()

    #### Visualize loss
    # from src.visualizations.loss_archetype_plot import loss_archetype_plot
    # import matplotlib.pyplot as plt
    # loss_archetype_plot(K_list=[1,2,3,4,5,6,7,8,9,10,20,30,40,50],
    #                     results_path='synthetic_results/ESS8_GB_results_2/All_AA_results.json')
    # plt.show()
    

    
    



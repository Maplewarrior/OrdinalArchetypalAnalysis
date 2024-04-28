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

    naive_data_params = {'N':1000, # irrelevant if load_data = True
                        'M':20, # irrelevant if load_data = True
                        'K':3, 
                        'p':5, 
                        'rb': True,
                        'mute': True,
                        'savefolder': 'naive_results',
                        'load_data': True,
                        'X_path': 'SyntheticData/1000_respondents/Data_naive_large.npy',
                        'Z_path': 'SyntheticData/1000_respondents/Data_naive_large.npy',
                        'A_path': 'SyntheticData/1000_respondents/Data_naive_large.npy'}

    naive_OSM_data_params = {'N':1000, # irrelevant if load_data = True
                            'M':20, # irrelevant if load_data = True
                            'K':3, 
                            'p':5, 
                            'rb': True,
                            'mute': True,
                            'savefolder': 'naive_OSM_results',
                            'load_data': True,
                            'X_path': 'SyntheticData/1000_respondents/OSM/data_naive_OSM_large.csv',
                            'Z_path': 'SyntheticData/1000_respondents/Data_naive_large.npy',
                            'A_path': 'SyntheticData/1000_respondents/Data_naive_large.npy'}

    naive_corrupted_data_params = {'N':1000, # irrelevant if load_data = True
                                  'M':20, # irrelevant if load_data = True
                                  'K':3, 
                                  'p':5, 
                                  'rb': True,
                                  'mute': True,
                                  'savefolder': 'naive_corrupted_results',
                                  'load_data': True,
                                  'X_path': 'SyntheticData/1000_respondents/Data_naive_large_corrupted.npz',
                                  'Z_path': 'SyntheticData/1000_respondents/Data_naive_large.npy',
                                  'A_path': 'SyntheticData/1000_respondents/Data_naive_large.npy'}

    naive_OSM_corrupted_data_params = {'N':1000, # irrelevant if load_data = True
                                       'M':20, # irrelevant if load_data = True
                                       'K':3, 
                                       'p':5, 
                                       'rb': True,
                                       'mute': True,
                                       'savefolder': 'naive_OSM_corrupted_results',
                                       'load_data': True,
                                       'X_path': 'SyntheticData/1000_respondents/OSM/data_naive_OSM_large_corrupted.csv',
                                       'Z_path': 'SyntheticData/1000_respondents/Data_naive_large.npy',
                                       'A_path': 'SyntheticData/1000_respondents/Data_naive_large.npy'}

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
    
    TEST_DATA_PARAMS = {'N':1000, # irrelevant if load_data = True
                            'M':20, # irrelevant if load_data = True
                            'K':3, 
                            'p':5, 
                            'rb': True,
                            'mute': True,
                            'savefolder': 'TestMHA',
                            'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                            'X_path': None, #'SyntheticData/1000_respondents/X.npy',
                            'Z_path': None, #'SyntheticData/1000_respondents/Z.npy',
                            'A_path': None #'SyntheticData/1000_respondents/A.npy'
                            }

    model_options = {'method': ['OAA', 'RBOAA'],
                    'with_init': [True],
                    'alternating': [False],
                    'beta_reg': [True],
                    'n_archetypes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                     20, 30, 40, 50]}
    # model_options_OSM = {'method': [''],
    #                     'with_init': [True],
    #                     'alternating': [False],
    #                     'beta_reg': [True],
    #                     'n_archetypes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #                                     20, 30, 40, 50]}
    model_options_test = {'method': ['OAA', 'RBOAA'],
                        'with_init': [True],
                        'alternating': [False],
                        'beta_reg': [True],
                        'n_archetypes': [1, 2, 3, 4, 5]}
    
    lrs = {} # hard coded for now
    RM = ResultMaker(data_params=ESS8_data_params, model_options=model_options, lrs=lrs)
    # RM.get_results()
    RM.get_ESS8_results()
    # ALL_DATA_PARAMS = [complex_data_params, complex_corrupted_data_params, naive_data_params, naive_corrupted_data_params]
    # for d_params in ALL_DATA_PARAMS:
    #     print("\n\nRunning analysis with: ", d_params['savefolder'],"\n\n")
    #     RM = ResultMaker(data_params=d_params, model_options=model_options, lrs=lrs)
    #     RM.get_results()

if __name__ == '__main__':
    # from src.utils.filter_ESS8 import filter_ESS8_data
    # df = filter_ESS8_data('RealData/ESS8_data.csv', only_GB=False)
    # filepath = "results/test_runs.json"
    # main(filepath=filepath)

    
    main()
    
    
    # import numpy as np
    # A = np.array([1, 2, 3])
    # L = list(A)
    # pdb.set_trace()
    # df = mean_across_runs('synthetic_results/naive_results/All_AA_results.json')
    # df = mean_across_runs('synthetic_results/1000_all_combinations.json')
    # print(df)
    
    # data = np.load('SyntheticData/1000_respondents/Data_naive_large_corrupted.npz')
    # X = data['arr_0']
    
    # tmp = np.load('SyntheticData/1000_respondents/Data_naive_large.npy', allow_pickle=False)
    # t = tmp.tolist()
    # df_naive = pd.read_csv('SyntheticData/1000_respondents/OSM/data_naive_OSM_large.csv', index_col=0)
    # df_naive_c = pd.read_csv('SyntheticData/1000_respondents/OSM/data_naive_OSM_large_corrupted.csv', index_col=0)

    # df_complex = pd.read_csv('SyntheticData/1000_respondents_complex/OSM/data_complex_OSM_large.csv', index_col=0)
    # df_complex_c = pd.read_csv('SyntheticData/1000_respondents_complex/OSM/data_complex_OSM_large_corrupted.csv', index_col=0)
    # import multiprocessing
    # print(multiprocessing.cpu_count())
    # # pdb.set_trace()



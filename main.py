import json
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
    means = df_res.groupby(by=['method', 'with_init', 'alternating', 'beta_reg']).mean()
    return df_res.loc[(df_res['alternating'] == False) & (df_res['method'] == 'RBOAA') & (df_res['with_init'] == True) & (df_res['beta_reg'] == True)]

    # means = df_res.groupby(by=['method', 'with_init', 'beta_reg', 'alternating']).mean()



def main():
    data_params = {'N':1000, # irrelevant if load_data = True
                   'M':20, # irrelevant if load_data = True
                   'K':3, 
                   'p':5, 
                   'rb': True,
                   'mute': True,
                   'load_data': False, # whether to load X, A and Z matrices from below paths or to generate new synthetic data.
                   'X_path': 'SyntheticData/synthetic_data_naive_small.pkl', #'SyntheticData/1000_respondents/X.npy',
                   'Z_path': 'SyntheticData/synthetic_data_naive.pkl', #'SyntheticData/1000_respondents/Z.npy',
                   'A_path': None #'SyntheticData/1000_respondents/A.npy'
                   }

    # model_options = {'method': ['OAA', 'RBOAA'],
    #                 'with_init': [False, True],
    #                 'alternating': [False, True],
    #                 'beta_reg': [False, True]}
    model_options = {'method': ['RBOAA', 'OAA'],
                    'with_init': [True],
                    'alternating': [False],
                    'beta_reg': [True],
                    'n_archetypes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                     20, 30, 40, 50]}
    lrs = {} # hard coded for now
    RM = ResultMaker(data_params=data_params, model_options=model_options, lrs=lrs)
    RM.get_results()

if __name__ == '__main__':
    # filepath = "results/test_runs.json"
    # main(filepath=filepath)
    main()
    # import numpy as np
    # A = np.array([1, 2, 3])
    # L = list(A)
    # pdb.set_trace()
    # df = mean_across_runs('synthetic_results/50_all_combinations.json')
    # df = mean_across_runs('synthetic_results/1000_all_combinations.json')
    # print(df)



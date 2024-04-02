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
    means = df_res.groupby(by=['method', 'with_init', 'beta_reg', 'alternating']).mean()


def main():
    data_params = {'N':800, 
                   'M':12, 
                   'K':4, 
                   'p':6, 
                   'rb': True,
                   'mute': True}

    model_options = {'method': ['OAA', 'RBOAA'],
                    'with_init': [False, True],
                    'alternating': [False, True],
                    'beta_reg': [False, True]}
    lrs = {} # hard coded for now
    RM = ResultMaker(data_params=data_params, model_options=model_options, lrs=lrs)
    RM.get_results()




if __name__ == '__main__':
    # filepath = "results/test_runs.json"
    # main(filepath=filepath)
    main()



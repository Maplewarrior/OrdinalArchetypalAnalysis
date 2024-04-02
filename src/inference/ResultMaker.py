import numpy as np
import json
import pdb

from src.AAM import AA as AA_class
from src.utils.eval_measures import NMI, MCC
from src.methods.CAA_class import _CAA
from src.methods.OAA_class import _OAA
from src.methods.RBOAA_class import _RBOAA

### Import old methods to allow for comparison with 4. sem implementation
from src.methods.OAA_class_old import _OAA as _OAA_old
from src.methods.RBOAA_class_old import _RBOAA as _RBOAA_ol

import multiprocessing

data_params = {'N':800, 
               'M':12, 
               'K':4, 
               'p':6, 
               'a_param':0.85,
               'b_param':5, 
               'sigma':-3, 
               'sigma_dev': 0.05,
               'rb': True,
               'mute': True}

model_options = {'method': ['OAA', 'RBOAA'],
                 'with_init': [False, True],
                  'alternating': [False],
                 'beta_reg': [False]}
                #  'alternating': [False, True],
                #  'beta_reg': [False, True]}

class ResultMaker:
    def __init__(self, data_params: dict, model_options: dict, lrs: dict) -> None:
        ### set global parameters
        self.data_params = data_params
        self.model_options = model_options
        self.n_repeats = 5
        self.CAA_lr = 0.01 # lr for CAA
        self.OAA_lr = 0.001 # lr for OAA and RBOAA
        self.n_iter = 20000 # max iterations in case early stopping does not take into effect
        self.early_stopping=True # early stopping if converged
        ### create empty result container
        self.results_init = {'method': [], 'with_init': [], 'beta_reg': [],
                        'alternating': [], 'NMI': [], 'MCC': []}

    def update_results(self, results, run):
        assert sorted(run.keys()) == sorted(results.keys()), 'Missing specifications for storing results!'
        for key, val in run.items():
            results[key].append(val)

        return results
    
    def make_synthetic_data(self, **kwargs):
        AA = AA_class()
        # print("\n\nKWARGS:")
        # for k, v in kwargs.items():
        #     print(k, v)
        AA.create_synthetic_data(kwargs['N'], kwargs['M'], kwargs['K'], kwargs['p'], kwargs['sigma'], kwargs['rb'], kwargs['b_param'], kwargs['a_param'], kwargs['sigma_dev'],mute=kwargs['mute'])
        self.columns = AA._synthetic_data.columns
        self._X = AA._synthetic_data.X # extract synthetic responses
        self._Z = AA._synthetic_data.Z
        self._A = AA._synthetic_data.A
        
    def make_analysis(self, results, run_specs):
        if run_specs['method'] == 'OAA':
            OAA = _OAA()
            OAA_res = OAA._compute_archetypes(self._X, self.data_params['K'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=True, with_CAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping)
            _NMI = NMI(OAA_res.A, self._A)
            _MCC = MCC(OAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            self.update_results(results, run_specs) # update results
            
        elif run_specs['method'] == 'RBOAA':
            RBOAA = _RBOAA()
            # run_specs = {'method': 'RBOAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            RBOAA_res = RBOAA._compute_archetypes(self._X, self.data_params['K'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=True, with_OAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, backup_itterations=True)
            _NMI = NMI(RBOAA_res.A, self._A)
            _MCC = MCC(RBOAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            self.update_results(results, run_specs) # update results
        
        else: # do CAA analysis (has no tunable parameters)
            CAA = _CAA()
            # run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            CAA_res = CAA._compute_archetypes(X=self._X, K=self.data_params['K'], p=self.data_params['p'], n_iter=self.n_iter, lr=self.CAA_lr, mute=True, 
                                    early_stopping=self.early_stopping, columns=self.columns, with_synthetic_data=True)
            _NMI = NMI(CAA_res.A, self._A)
            _MCC = MCC(CAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            self.update_results(results, run_specs)
    
    def result_helper(self, data_params: list):
        """
        Creates results for all combinations of hyperparameters specified.
        """
        sigma = data_params[0]
        a_param = data_params[1]
        b_param = data_params[2]
        sigma_dev = data_params[3]
        ### create synthetic data
        self.make_synthetic_data(a_param=a_param, b_param=b_param, sigma=sigma, sigma_dev=sigma_dev, **self.data_params)
        results = self.results_init.copy()
        for method in self.model_options['method']:
            for beta_reg in model_options['beta_reg']:
                for alternating in model_options['alternating']:
                    for _init in model_options['with_init']:
                        print(f'Doing analysis! Params are: {sigma} {a_param} {b_param} {sigma_dev}')
                        for _ in range(self.n_repeats):
                            run_specs = {'method': method, 'with_init': _init, 'beta_reg': beta_reg, 'alternating': alternating, 'MCC': None, 'NMI': None}
                            try:
                                self.make_analysis(results, run_specs=run_specs)
                            except Exception as e:
                                print(f"Error occured which reads: {e}\nThe specs were: {run_specs}")
        ### get CAA results
        for _ in range(self.n_repeats):
            self.make_analysis({'method': 'CAA'})
        
        with open(f'results/result_sigma={sigma}_a={a_param}_b={b_param}_dev={sigma_dev}.json', 'w') as f:
            json.dump(self.results, f)
    
    def get_results(self):
        a_params = [1, 0.85] #[0.85, 1, 2]
        b_params = [5, 10] #[1, 5, 10]
        sigmas = [-3.] #, -1.5078, -1.0502]
        sigma_stds = [0.01]
        all_data_params = []

        for sigma in sigmas:
            for sigma_std in sigma_stds:
                for a_param in a_params:
                    for b_param in b_params:
                        all_data_params.append([sigma, a_param, b_param, sigma_std])
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
            p.map(self.result_helper, all_data_params)


if __name__ == '__main__':
    pass
    # X, Z, A = get_synthetic_data(**data_params)
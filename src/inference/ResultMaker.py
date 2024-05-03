import numpy as np
import json
import os
import pdb
import pickle
import pandas as pd

from src.utils.filter_ESS8 import filter_ESS8_data

from src.AAM import AA as AA_class
from src.utils.eval_measures import NMI, MCC
from src.methods.CAA_class import _CAA
from src.methods.OAA_class import _OAA
from src.methods.RBOAA_class import _RBOAA

### Import old methods to allow for comparison with 4. sem implementation
from src.methods.OAA_class_old import _OAA as _OAA_old
from src.methods.RBOAA_class_old import _RBOAA as _RBOAA_old


import multiprocessing
class ResultMaker:
    def __init__(self, data_params: dict, model_options: dict, lrs: dict) -> None:
        ### set global parameters
        self.data_params = data_params
        self.model_options = model_options
        self.n_repeats = 10
        self.CAA_lr = 0.1 # lr for CAA
        self.OAA_lr = 0.01 # lr for OAA and RBOAA
        self.n_iter = 20000 # max iterations in case early stopping does not take into effect
        self.early_stopping=True # early stopping if converged
        ### create empty result container
        self.results_init = {'method': [], 'with_init': [], 'beta_reg': [],
                             'alternating': [], 'n_archetypes': [],
                             'NMI': [], 'MCC': [], 'loss': []}
        self.results_ESS8_init = {'method': [], 'with_init': [], 'beta_reg': [],
                             'alternating': [], 'n_archetypes': [],
                             'loss': []}
        
        savefolder = self.data_params['savefolder']
        self.savedir = f'synthetic_results/{savefolder}'
        self.data_savedir = f'SyntheticData/{savefolder}'

        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        
    def update_results(self, results, run):
        assert sorted(run.keys()) == sorted(results.keys()), 'Missing specifications for storing results!'
        for key, val in run.items():
            results[key].append(val)

        return results
    
    def save_result_obj(self, res_obj, repeat_num: int, n_archetypes: int):
        if not os.path.exists(f'{self.savedir}/{res_obj.type}_objects'):
            os.mkdir(f'{self.savedir}/{res_obj.type}_objects')
        
        with open(f'{self.savedir}/{res_obj.type}_objects/{res_obj.type}_K={n_archetypes}_rep={repeat_num}', 'wb') as f:
            pickle.dump(res_obj, f, pickle.HIGHEST_PROTOCOL)
        
    def load_data(self, X_path: str, Z_path: str, A_path: str):
        _, X_ext = os.path.splitext(X_path)
        
        if X_ext == '.npy':
            try: # complex data
                self._X = np.load(X_path)

            except ValueError: # Naive data stored as pickle object in .npy file
                tmp = np.load(X_path, allow_pickle=True)
                t = tmp.tolist()
                self._X = t.X

            self.load_AZ_npy(Z_path, A_path)

        elif X_ext == '.npz': # Naive/complex corrupted 
            data = np.load(X_path)
            self._X = data['arr_0']
            self.load_AZ_npy(Z_path, A_path) # complex + naive handled

        elif X_ext == '.csv': # complex/naive OSM + OSM corrupted
            X = pd.read_csv(X_path, index_col=0).values
            self._X = X.T if X.shape[0] > X.shape[1] else X # M x N matrix
            self.columns = [f'q{i}' for i in range(self._X.shape[0])]
            self.load_AZ_npy(Z_path, A_path)

        elif X_path == 'ESS8_GB': # load ESS8 subset 
            self._X, self.columns = filter_ESS8_data('RealData/ESS8_data.csv', only_GB=True)

        elif X_path == 'ESS8': # load full ESS8 dataset
            self._X, self.columns = filter_ESS8_data('RealData/ESS8_data.csv', only_GB=False)
        
        elif X_ext == '.pkl':
            raise NotImplementedError()
    
    def load_AZ_npy(self, Z_path: str, A_path:str):
        try: # for complex
            self._Z = np.load(Z_path)
            self._A = np.load(A_path)

        except ValueError: # for naive
            tmp = np.load(Z_path, allow_pickle=True)
            t = tmp.tolist()
            self._A = t.A
            self._Z = t.Z
    
    def make_synthetic_data(self, **kwargs):
        AA = AA_class()
        if kwargs['load_data'] == False:
            AA.create_synthetic_data(kwargs['N'], kwargs['M'], kwargs['K'], kwargs['p'], kwargs['sigma'], kwargs['rb'], kwargs['b_param'], kwargs['a_param'], kwargs['sigma_dev'],mute=kwargs['mute'])
            self.columns = AA._synthetic_data.columns
            self._X = AA._synthetic_data.X # extract synthetic responses
            self._Z = AA._synthetic_data.Z # extract archetype matrix
            self._A = AA._synthetic_data.A # extract weighting matrix
            ### Save synthetic data

            dir = self.data_savedir
            if not os.path.exists(dir):
                os.mkdir(dir)
            
            np.save(f'{dir}/X.npy', self._X)
            np.save(f'{dir}/Z.npy',self._Z)
            np.save(f'{dir}/A.npy',self._A)
            with open(f'{dir}/data_parameters.json', 'w') as f:
                json.dump(kwargs, f)
            pdb.set_trace()

        else:
            self.load_data(kwargs['X_path'], kwargs['Z_path'], kwargs['A_path'])
        print("X", self._X.shape)
        # print("A", self._A.shape)
        # print("Z", self._Z.shape)
        # assert list(self._X.shape) == [20, 1000], 'Shape mismatch for X!'
        # assert list(self._Z.shape) == [20, 3], 'Shape mismatch for X!'
        # assert list(self._A.shape) == [3, 1000], 'Shape mismatch for X!'
        # self.columns = [f'q{i}' for i in range(1, self._X.shape[0]+1)]
        
    
    def make_analysis(self, results, run_specs, repeat_num: int):
        if run_specs['method'] == 'OAA':
            ## run old OAA analysis
            # OAA_old = _OAA_old()
            # OAA_old_res = OAA_old._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=True,columns=self.columns, early_stopping=self.early_stopping)
            # NMI_old = NMI(OAA_old_res.A, self._A)
            # MCC_old = MCC(OAA_old_res.Z, self._Z)
            # old_run_specs = {'method': 'OAA_old', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': MCC_old, 'NMI': NMI_old}
            # self.update_results(results, old_run_specs)
            
            ### run OAA analysis
            OAA = _OAA()
            # OAA_res = OAA._compute_archetypes(self._X, run_spects['n_archetypes], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_CAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping)
            
            ### Hard code OAA_res to never do a CAA init
            OAA_res = OAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_CAA_initialization=False,columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, seed=repeat_num)
            run_specs['with_init'] = False
            
            _NMI = NMI(OAA_res.A, self._A)
            _MCC = MCC(OAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            run_specs['loss'] = list(OAA_res.loss)
            self.save_result_obj(OAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results
            
        elif run_specs['method'] == 'RBOAA':
            ### run old RBOAA analysis
            # RBOAA_old = _RBOAA_old()
            # RBOAA_old_res = RBOAA_old._compute_archetypes(self._X, self.data_params['K'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=True, with_OAA_initialization=True, early_stopping=True, columns=self.columns)
            # NMI_old = NMI(RBOAA_old_res.A, self._A)
            # MCC_old = MCC(RBOAA_old_res.Z, self._Z)
            # old_run_specs = {'method': 'RBOAA_old', 'with_init': True, 'beta_reg': False, 'alternating': False, 'MCC': MCC_old, 'NMI': NMI_old}
            # self.update_results(results, old_run_specs)

            ### run RBOAA analysis
            RBOAA = _RBOAA()
            # run_specs = {'method': 'RBOAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            RBOAA_res = RBOAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_OAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, backup_itterations=True, seed=repeat_num)
            _NMI = NMI(RBOAA_res.A, self._A)
            _MCC = MCC(RBOAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            run_specs['loss'] = list(RBOAA_res.loss)
            self.save_result_obj(RBOAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results

        elif run_specs['method'] == 'CAA': # do CAA analysis (has no tunable parameters)
            CAA = _CAA()
            # run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            CAA_res = CAA._compute_archetypes(X=self._X, K=run_specs['n_archetypes'], p=self.data_params['p'], n_iter=self.n_iter, lr=self.CAA_lr, mute=self.data_params['mute'], 
                                              early_stopping=self.early_stopping, columns=self.columns, with_synthetic_data=True, seed=repeat_num)
            _NMI = NMI(CAA_res.A, self._A)
            _MCC = MCC(CAA_res.Z, self._Z)
            run_specs['NMI'] = _NMI
            run_specs['MCC'] = _MCC
            run_specs['loss'] = list(CAA_res.loss)
            self.save_result_obj(CAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs)
        
        else:
            return
        
    def result_helper(self, hyperparams: list):
        """
        Creates results for all combinations of model hyperparameters specified.
        """
        ### Exctract synthetic data parameters
        sigma = hyperparams[0]
        a_param = hyperparams[1]
        b_param = hyperparams[2]
        sigma_dev = hyperparams[3]
        ### create synthetic data
        self.make_synthetic_data(a_param=a_param, b_param=b_param, sigma=sigma, sigma_dev=sigma_dev, **self.data_params)
        results = self.results_init.copy()
        for method in self.model_options['method']:
            for beta_reg in self.model_options['beta_reg']:
                for alternating in self.model_options['alternating']:
                    for _init in self.model_options['with_init']:
                        for K in self.model_options['n_archetypes']:
                            print(f'Doing analysis! Params are: {sigma} {a_param} {b_param} {sigma_dev}')
                            for rep in range(self.n_repeats):
                                run_specs = {'method': method, 'with_init': _init, 'beta_reg': beta_reg, 'alternating': alternating, 'n_archetypes': K, 'MCC': None, 'NMI': None, 'loss': None}
                                try:
                                    self.make_analysis(results, run_specs=run_specs, repeat_num=rep)
                                except Exception as e:
                                    print(f"Error occured which reads: {e}\nThe specs were: {run_specs}")
        ### get CAA results
        for K in self.model_options['n_archetypes']:
            for rep in range(self.n_repeats):
                run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'n_archetypes': K, 'MCC': None, 'NMI': None, 'loss': None}
                self.make_analysis(results, run_specs, repeat_num=rep)
        
        ### save loss, NMI and MCC results
        with open(f'{self.savedir}/All_AA_results.json', 'w') as f:
            json.dump(results, f)
        
    def get_results(self):
        a_params = [1.]
        b_params = [1.5]
        sigmas = [-9.21] #, -1.5078, -1.0502]
        sigma_stds = [1e-6]
        all_data_params = []

        for sigma in sigmas:
            for sigma_std in sigma_stds:
                for a_param in a_params:
                    for b_param in b_params:
                        all_data_params.append([sigma, a_param, b_param, sigma_std])
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        # with multiprocessing.Pool(multiprocessing.cpu_count()//4) as p:
            p.map(self.result_helper, all_data_params)

    
    def make_ESS8_analysis(self, results, run_specs, repeat_num: int):
        if run_specs['method'] == 'OAA':
            ### run OAA analysis
            OAA = _OAA()
            # OAA_res = OAA._compute_archetypes(self._X, run_spects['n_archetypes], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_CAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping)
            
            ### Hard code OAA_res to never do a CAA init
            OAA_res = OAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_CAA_initialization=False,columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, seed=repeat_num)
            run_specs['with_init'] = False
            run_specs['loss'] = list(OAA_res.loss)
            self.save_result_obj(OAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results
            
        elif run_specs['method'] == 'RBOAA':
            ### run RBOAA analysis
            RBOAA = _RBOAA()
            # run_specs = {'method': 'RBOAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            RBOAA_res = RBOAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_OAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, backup_itterations=True, seed=repeat_num)
            run_specs['loss'] = list(RBOAA_res.loss)
            self.save_result_obj(RBOAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results

        elif run_specs['method'] == 'CAA': # do CAA analysis (has no tunable parameters)
            CAA = _CAA()
            # run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            CAA_res = CAA._compute_archetypes(X=self._X, K=run_specs['n_archetypes'], p=self.data_params['p'], n_iter=self.n_iter, lr=self.CAA_lr, mute=self.data_params['mute'], 
                                              early_stopping=self.early_stopping, columns=self.columns, with_synthetic_data=True, seed=repeat_num)
            run_specs['loss'] = list(CAA_res.loss)
            self.save_result_obj(CAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs)
        
        else:
            return
    
    def get_ESS8_results(self):
        results = self.results_ESS8_init.copy()
        self.load_data(self.data_params['X_path'], None, None)
        for method in self.model_options['method']:
            for beta_reg in self.model_options['beta_reg']:
                for alternating in self.model_options['alternating']:
                    for _init in self.model_options['with_init']:
                        for K in self.model_options['n_archetypes']:
                            print(f"Doing ESS8 analysis!")
                            for rep in range(self.n_repeats):
                                run_specs = {'method': method, 'with_init': _init, 'beta_reg': beta_reg, 'alternating': alternating, 'n_archetypes': K, 'loss': None}
                                try:
                                    self.make_ESS8_analysis(results, run_specs=run_specs, repeat_num=rep)
                                except Exception as e:
                                    print(f"Error occured which reads: {e}\nThe specs were: {run_specs}")
    
        for K in self.model_options['n_archetypes']:
            for rep in range(self.n_repeats):
                run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'n_archetypes': K, 'loss': None}
                self.make_ESS8_analysis(results, run_specs, repeat_num=rep)

                
            

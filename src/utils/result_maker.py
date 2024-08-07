import numpy as np
import json
import os
import pickle
import pandas as pd
import multiprocessing

from src.utils.filter_ESS8 import filter_ESS8_data
from src.utils.corruptData import corruptData
from src.utils.synthetic_data_class import _synthetic_data
# from src.utils.synthetic_data_naive import synthetic_data_naive
# from deprecated.AAM import AA as AA_class # TODO: Update this and synthetic data generation
from src.utils.eval_measures import NMI, MCC
from src.methods.CAA_class import _CAA
from src.methods.OAA_class import _OAA
from src.methods.RBOAA_class import _RBOAA


### Import old methods to allow for comparison with 4. sem implementation
# from deprecated.OAA_class_old import _OAA as _OAA_old
# from deprecated.RBOAA_class_old import _RBOAA as _RBOAA_old


class ResultMaker:
    def __init__(self, cfg: dict) -> None:
        ### set global parameters
        self.CFG = cfg
        self.n_repeats = self.CFG['training']['n_repeats']
        self.CAA_lr = self.CFG['training']['CAA_lr']
        self.OAA_lr = self.CFG['training']['OAA_lr']
        self.RBOAA_lr = self.CFG['training']['RBOAA_lr']
        self.n_iter = self.CFG['training']['n_iter']
        self.early_stopping=self.CFG['training']['early_stopping']
        self.mute = self.CFG['training']['mute']
        self.p = self.CFG['data']['p'] if not self.CFG['data']['use_synthetic_data'] else self.CFG['data']['synthetic_data_params']['p']

        ### create empty result container
        self.results_init = {'method': [], 'with_init': [], 'beta_reg': [],
                             'alternating': [], 'n_archetypes': [],
                             'NMI': [], 'MCC': [], 'loss': []}
        
        self.results_ESS8_init = {'method': [], 'with_init': [], 'beta_reg': [],
                                  'alternating': [], 'n_archetypes': [],
                                  'loss': []}
        
        # savefolder = self.data_params['savefolder']
        self.savedir = self.CFG['data']['results']['checkpoint_dir'] #f'synthetic_results/{savefolder}'
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
       
        # store experiment configuration for later reference
        with open(f"{self.savedir}/experiment_config.json", 'w') as fp:
            json.dump(self.CFG, fp)
        
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
            self._X = np.load(X_path)
            # self.load_AZ_npy(Z_path, A_path)

        elif X_ext == '.npz': # Naive/complex corrupted
            data = np.load(X_path)
            self._X = data['arr_0']
            # self.load_AZ_npy(Z_path, A_path) # complex + naive handled

        elif X_ext == '.csv': # complex/naive OSM + OSM corrupted
            X = pd.read_csv(X_path, index_col=0).values
            self._X = X.T if X.shape[0] > X.shape[1] else X # M x N matrix
            self.columns = [f'q{i}' for i in range(self._X.shape[0])]
            # self.load_AZ_npy(Z_path, A_path)
        
    def load_OSM_data(self, X_OSM_path: str):
        X = pd.read_csv(X_OSM_path, index_col=0).values
        self._X_OSM = X.T if X.shape[0] > X.shape[1] else X # M x N matrix
        # self.columns = [f'q{i}' for i in range(self._X.shape[0])]

    # def load_AZ_npy(self, Z_path: str, A_path:str):
    #     if Z_path is None:
    #         return
    #     try: # for complex
    #         self._Z = np.load(Z_path)
    #         self._A = np.load(A_path)

    #     except ValueError: # for naive data
    #         tmp = np.load(Z_path, allow_pickle=True)
    #         t = tmp.tolist()
    #         self._A = t.A
    #         self._Z = t.Z
    
    def make_data(self, **kwargs):
        
        if not self.CFG['data']['use_synthetic_data']:
            Z_path = None if 'Z_path' not in kwargs.keys() else kwargs['Z_path'] # extract ground truth archetype matrix if specified.
            A_path = None if 'A_path' not in kwargs.keys() else kwargs['A_path'] # extract ground truth respondent weighting if speficied.
            self.load_data(kwargs['X_path'], A_path=A_path, Z_path=Z_path)
        
        else: # use synthetic data
            syn_data = _synthetic_data(kwargs['N'], kwargs['M'], kwargs['K'], kwargs['p'], kwargs['sigma'], kwargs['rb'], kwargs['a_param'], kwargs['b_param'], kwargs['sigma_dev'])#,mute=kwargs['mute'])
            # AA.create_synthetic_data(kwargs['N'], kwargs['M'], kwargs['K'], kwargs['p'], kwargs['sigma'], kwargs['rb'], kwargs['b_param'], kwargs['a_param'], kwargs['sigma_dev'],mute=kwargs['mute'])
            self.columns = syn_data.columns
            self._X = syn_data.X # extract synthetic responses
            self._Z = syn_data.Z # extract archetype matrix
            self._A = syn_data.A # extract weighting matrix
        
        if self.CFG['data']['OSM_data_path'] is not None:
            self.load_OSM_data(self.CFG['data']['OSM_data_path'])

        if self.CFG['data']['do_corrupt']:
            self._X_org = self._X.copy()
            self._X, _, _ = corruptData(data=self._X_org.copy(), corruption_rate=self.CFG['data']['p_corrupt'], likertScale=self.p)

            if self.CFG['data']['OSM_data_path'] is not None:
                self._X_OSM_org = self._X_OSM.copy()
                self._X_OSM, _, _ = corruptData(data=self._X_OSM_org.copy(), corruption_rate=self.CFG['data']['p_corrupt'], likertScale=self.p)

        # save the experiment data for later reference
        if not os.path.exists(f'{self.savedir}/experiment_data'):
            os.mkdir(f'{self.savedir}/experiment_data')
    
        if hasattr(self, '_X_org'):
            pd.DataFrame(self._X_org).to_csv(f'{self.savedir}/experiment_data/X.csv', index=False)
            pd.DataFrame(self._X).to_csv(f'{self.savedir}/experiment_data/X_cor.csv', index=False)
        else:
            pd.DataFrame(self._X).to_csv(f'{self.savedir}/experiment_data/X.csv', index=False)

        # if hasattr(self, '_X_OSM'):
        #     pd.DataFrame(self._X_OSM).to_csv(f'{self.savedir}/experiment_data/X_OSM.csv', index=False)
        
        # if hasattr(self, '_X_OSM_org'):
        #     pd.DataFrame(self._X).to_csv(f'{self.savedir}/experiment_data/X_cor.csv', index=False)

        if not hasattr(self, 'columns'):
            self.columns = [f'q{i}' for i in range(1, self._X.shape[0]+1)]

    def make_analysis(self, results, run_specs, repeat_num: int):
        if run_specs['method'] == 'OAA':            
            ### run OAA analysis
            OAA = _OAA()
            OAA_res = OAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.p, self.n_iter, self.OAA_lr, mute=self.mute, with_CAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping)
            
            ### Hard code OAA_res to never do a CAA init
            # OAA_res = OAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.data_params['p'], self.n_iter, self.OAA_lr, mute=self.data_params['mute'], with_CAA_initialization=False,columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, seed=repeat_num)
            run_specs['with_init'] = False
            if hasattr(self, '_A') and hasattr(self, '_Z'):
                run_specs = self.calc_NMI_and_MCC(OAA_res, self._A, self._Z, run_specs)
            
            run_specs['loss'] = list(OAA_res.loss)
            self.save_result_obj(OAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results
            
        elif run_specs['method'] == 'RBOAA':
            ### run RBOAA analysis
            RBOAA = _RBOAA()
            RBOAA_res = RBOAA._compute_archetypes(self._X, run_specs['n_archetypes'], self.p, self.n_iter, self.OAA_lr, mute=self.mute, with_OAA_initialization=run_specs['with_init'],columns=self.columns, alternating=run_specs['alternating'], beta_regulators=run_specs['beta_reg'], early_stopping=self.early_stopping, backup_itterations=True, seed=repeat_num)
            if hasattr(self, '_A') and hasattr(self, '_Z'):
                run_specs = self.calc_NMI_and_MCC(RBOAA_res, self._A, self._Z, run_specs)
            run_specs['loss'] = list(RBOAA_res.loss)
            self.save_result_obj(RBOAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs) # update results

        elif run_specs['method'] == 'CAA': # do CAA analysis (has no tunable parameters)
            CAA = _CAA()
            # run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            CAA_res = CAA._compute_archetypes(X=self._X, K=run_specs['n_archetypes'], p=self.p, n_iter=self.n_iter, lr=self.CAA_lr, mute=self.mute, 
                                              early_stopping=self.early_stopping, columns=self.columns, with_synthetic_data=True, seed=repeat_num)
            
            if hasattr(self, '_A') and hasattr(self, '_Z'):
                run_specs = self.calc_NMI_and_MCC(CAA_res, self._A, self._Z, run_specs)
            
            run_specs['loss'] = list(CAA_res.loss)
            self.save_result_obj(CAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs)
        
        elif run_specs['method'] == 'TSAA':
            TSAA = _CAA()
            # run_specs = {'method': 'CAA', 'with_init': False, 'beta_reg': False, 'alternating': False, 'MCC': None, 'NMI': None}
            TSAA_res = TSAA._compute_archetypes(X=self._X_OSM, K=run_specs['n_archetypes'], p=self.p, n_iter=self.n_iter, lr=self.CAA_lr, mute=self.mute, analysis_type='TSAA',
                                                early_stopping=self.early_stopping, columns=self.columns, with_synthetic_data=True, seed=repeat_num)
            
            if hasattr(self, '_A') and hasattr(self, '_Z'):
                run_specs = self.calc_NMI_and_MCC(TSAA_res, self._A, self._Z, run_specs)
            
            run_specs['loss'] = list(TSAA_res.loss)
            self.save_result_obj(TSAA_res, repeat_num, run_specs['n_archetypes'])
            self.update_results(results, run_specs)
        
        else:
            return

    def calc_NMI_and_MCC(self, result_obj, A_true, Z_true, run_specs):
        _NMI = NMI(result_obj.A, self._A)
        _MCC = MCC(result_obj.Z, self._Z)
        run_specs['NMI'] = _NMI
        run_specs['MCC'] = _MCC
        return run_specs

    def result_helper(self, hyperparams: list):
        """
        Creates results for all combinations of model hyperparameters specified.
        """
        ### Exctract synthetic data parameters
        data_params = {'sigma': hyperparams[0],
                       'a_param': hyperparams[1],
                       'b_param': hyperparams[2],
                       'sigma_dev': hyperparams[3],
                       'N': self.CFG['data']['synthetic_data_params']['N'],
                       'M': self.CFG['data']['synthetic_data_params']['M'],
                       'K': self.CFG['data']['synthetic_data_params']['K'],
                       'p': self.CFG['data']['synthetic_data_params']['p'],
                       'rb': self.CFG['data']['synthetic_data_params']['rb']}
        
        ### create synthetic data
        self.make_data(**data_params)
        results = self.results_init.copy()

        ## get CAA/TSAA results
        for method in self.CFG['training']['parameter_tuning']['method']:
            if method not in ["CAA", "TSAA"]:
                continue
            for K in self.CFG['training']['parameter_tuning']['n_archetypes']:
                for rep in range(self.n_repeats):
                    run_specs = {'method': method, 'with_init': False, 'beta_reg': False, 'alternating': False, 'n_archetypes': K, 'loss': None}
                    self.make_analysis(results, run_specs, repeat_num=rep)
        
        ## get OAA/RBOAA results
        for method in self.CFG['training']['parameter_tuning']['method']:
            if method in ['CAA', 'TSAA']:
                continue
            for beta_reg in self.CFG['training']['parameter_tuning']['beta_regulators']:
                for alternating in self.CFG['training']['parameter_tuning']['alternating']:
                    for _init in self.CFG['training']['parameter_tuning']['with_init']:
                        for K in self.CFG['training']['parameter_tuning']['n_archetypes']:
                            # print(f'Doing synthetic analysis! Params are: {sigma} {a_param} {b_param} {sigma_dev}')
                            for rep in range(self.n_repeats):
                                run_specs = {'method': method, 'with_init': _init, 'beta_reg': beta_reg, 'alternating': alternating, 'n_archetypes': K, 'MCC': None, 'NMI': None, 'loss': None}
                                try:
                                    self.make_analysis(results, run_specs=run_specs, repeat_num=rep)
                                except Exception as e:
                                    print(f"Error occured which reads: {e}\nThe specs were: {run_specs}")
        
        ### save loss, NMI and MCC results
        with open(f'{self.savedir}/All_AA_results.json', 'w') as f:
            json.dump(results, f)
        
    def get_synthetic_results(self):
        a_params = self.CFG['data']['synthetic_data_params']['a_param']
        b_params = self.CFG['data']['synthetic_data_params']['b_param']
        sigmas = self.CFG['data']['synthetic_data_params']['sigma'] #[-9.21] #, -1.5078, -1.0502]
        sigma_stds = self.CFG['data']['synthetic_data_params']['sigma_dev']
        all_data_params = []
        for sigma in sigmas:
            for sigma_std in sigma_stds:
                for a_param in a_params:
                    for b_param in b_params:
                        all_data_params.append([sigma, a_param, b_param, sigma_std])
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        # with multiprocessing.Pool(multiprocessing.cpu_count()//4) as p:
            p.map(self.result_helper, all_data_params)
    
    def get_real_results(self):
        results = self.results_ESS8_init.copy()
        
        self.make_data(X_path=self.CFG['data']['input_data_path'])
        # self.make_data(**self.data_params)
        methods = self.CFG['training']['parameter_tuning']['method']
        print(f'METHODS: {methods}')
        for method in methods:
            if method not in ["CAA", "TSAA"]:
                continue
            for K in self.CFG['training']['parameter_tuning']['n_archetypes']:
                for rep in range(self.n_repeats):
                    run_specs = {'method': method, 'with_init': False, 'beta_reg': False, 'alternating': False, 'n_archetypes': K, 'loss': None}
                    self.make_analysis(results, run_specs, repeat_num=rep)
        for method in methods:
            if method in ["CAA", "TSAA"]:
                continue
            for beta_reg in self.CFG['training']['parameter_tuning']['beta_regulators']:
                for alternating in self.CFG['training']['parameter_tuning']['alternating']:
                    for _init in self.CFG['training']['parameter_tuning']['with_init']:
                        for K in self.CFG['training']['parameter_tuning']['n_archetypes']:
                            print(f"Doing ESS8 analysis!")
                            for rep in range(self.n_repeats):
                                run_specs = {'method': method, 'with_init': _init, 'beta_reg': beta_reg, 'alternating': alternating, 'n_archetypes': K, 'loss': None}
                                try:
                                    self.make_analysis(results, run_specs=run_specs, repeat_num=rep)
                                except Exception as e:
                                    print(f"Error occured which reads: {e}\nThe specs were: {run_specs}")
        

        ### save loss, NMI and MCC results
        with open(f'{self.savedir}/All_AA_results.json', 'w') as f:
            json.dump(results, f)
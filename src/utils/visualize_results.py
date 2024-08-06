import matplotlib
import os
import json
import numpy as np
import pandas as pd
from src.visualizations.functions import load_analyses
from src.visualizations.archetypal_answers import plot_archetypal_answers
from src.visualizations.loss_archetype_plot import loss_archetype_plot
from src.visualizations.NMI_archetypes import NMI_archetypes
from src.visualizations.NMI_stability import plot_NMI_stability
from src.visualizations.denoising import denoising
from src.visualizations.response_bias import response_bias_plot as rb_plot
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class VisualizeResult:
    def __init__(self, cfg: dict) -> None:
        self.CFG = cfg
        self.palette = {'RBOAA': "#EF476F", 'OAA': "#FFD166", 'AA': "#06D6A0","TSAA" : "#073B4C"}
        if not os.path.exists(self.CFG['visualization']['save_dir']):
            os.makedirs(self.CFG['visualization']['save_dir'])
        self.p = self.CFG['data']['p'] if not self.CFG['data']['use_synthetic_data'] else self.CFG['data']['synthetic_data_params']['p']
        self.__load_experiment()

    def __load_experiment(self):
        # load the saved result objects
        self.result_objects = load_analyses(analysis_dir=self.CFG['data']['results']['checkpoint_dir'])
        # load json file with loss dynamics and NMI calculations
        self.result_path = '{ckpt_path}/All_AA_results.json'.format(ckpt_path=self.CFG['data']['results']['checkpoint_dir'])
        if not os.path.exists(self.result_path):
            print(f'The path: {self.result_path} does not exist. This may be because the analysis terminated prior to being finished. As a result some plots will be unavaiable.')
        else:
            with open(self.result_path, 'r') as f:
                self.analysis_metadata = json.load(f)
        # methods = list(self.result_objects.keys())
        # n_archetypes = list(self.result_objects[methods[0]].keys())
        # n_reps = list(self.result_objects[methods[0]][n_archetypes[0]].keys())
        data_dir = "{ckpt_dir}/experiment_data".format(ckpt_dir=self.CFG['data']['results']['checkpoint_dir'])
        self.X = pd.read_csv(f"{data_dir}/X.csv").values
        if "X_cor.csv" in os.listdir(data_dir):
            self.X_cor = pd.read_csv(f"{data_dir}/X_cor.csv").values

    
    def loss_archetype_plot(self, TSOAA_result_path: str = None):
        loss_archetype_plot(K_list=self.CFG['training']['parameter_tuning']['n_archetypes'],
                            results_path=self.result_path,
                            results_path2=TSOAA_result_path,
                            methods_colors=self.palette,
                            savedir=self.CFG['visualization']['save_dir'])
    
    def NMI_stability_plot(self):
        plot_NMI_stability(folder_path=self.CFG['data']['results']['checkpoint_dir'],
                            K_list=self.CFG['training']['parameter_tuning']['n_archetypes'],
                            repetitions= self.CFG['training']['n_repeats'],
                            methods_colors=self.palette,
                            savedir=self.CFG['visualization']['save_dir'])    
        
    def NMI_archetype_plot(self, TSOAA_result_path: str = None):
        NMI_archetypes(K_list=self.CFG['training']['parameter_tuning']['n_archetypes'],
                       results_path=self.result_path,
                       results_path2=TSOAA_result_path,
                       methods_colors=self.palette,
                       savedir=self.CFG['visualization']['save_dir'])
    
    def archetypal_answers_plot(self, 
                           method: str, 
                           K: int, 
                           rep: int,
                           likert_text: list[str] = None, 
                           questions: list[str] = None,
                           type: str = 'lines'):
        
        if likert_text is None:
            likert_text = [i for i in range(1, len(np.unique(self.X))+1)]
        if questions is None:
            questions = [f'Q{i}' for i in range(1, self.X.shape[0]+1)] # TODO Double check that it is M x N in result object
        
        res_obj = self.result_objects[method][f'K{K}'][rep]
        Z = res_obj.X @ res_obj.B
        plot_archetypal_answers(res_obj.X, Z, self.p, likert_text, 
                                questions, startColor=self.palette[method], 
                                type = type, 
                                savepath=f"{self.CFG['visualization']['save_dir']}/{method}_archetypal_answers_plot.png")
    
    def denoising_plot(self):
        # denoising(dataObj,dataObjCorr,dataObjOSMCorr,K_list,p,figName)
        """
        Kræver original X, corrupted X og evt OSM corrupted X...
        """
        p = self.CFG['data']['p'] if not self.CFG['data']['use_synthetic_data'] else self.CFG['data']['synthetic_data_params']['p']
        denoising(self.X, 
                  self.X_cor, 
                  self.result_objects,
                  K_list=self.CFG['training']['parameter_tuning']['n_archetypes'],
                  p=p,
                  n_reps=self.CFG['training']['n_repeats'],
                  my_pallette=self.palette,
                  savedir=self.CFG['visualization']['save_dir'])

    def response_bias_plot(self, K:int, rep:int):
        # response_bias_plot(X, RBOAA_betasQ20, OAA_betasQ20,RBOAA_betasQ100, OAA_betasQ100,p, plotname, synthetic_betas=None)
        # TODO: Fix struktur generelt
        RBOAA_obj = self.result_objects['RBOAA'][f'K{K}'][rep]
        OAA_obj = self.result_objects['OAA'][f'K{K}'][rep]
        rb_plot(self.X, RBOAA_obj.b, OAA_obj.b, self.p, self.CFG['visualization']['save_dir'], my_pallette=self.palette, synthetic_betas=None)
        

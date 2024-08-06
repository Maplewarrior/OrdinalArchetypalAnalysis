"""
Moved to visualizations/functions
"""

# import pickle
# import re
# import os
# import json
# from src.utils.eval_measures import NMI, MCC

# def load_result_obj(path: str):
#     file = open(path,'rb')
#     object_file = pickle.load(file)
#     file.close()
#     return object_file

# def load_analyses(analysis_dir: str):
#     """
#     Function that loads results from a given analysis.
#     The format is a nested dictionary on the form results[AA_method][n_archetypes][repetition_num]
#     The result objects saved have all matrices and parameters inside them. E
#     """
#     # folder = f'C:/Users/aejew/Downloads/AA_results/AA_results/{analysis_dir}'
#     folder = f'synthetic_results/{analysis_dir}'
#     results = {'RBOAA': {}, 'OAA': {}, 'CAA': {}} if 'OSM' not in analysis_dir else {'TSAA': {}}

#     for method in results.keys():
#         method_dir = f'{folder}/{method}_objects'
#         all_files = os.listdir(method_dir)
#         for file in all_files:
#             obj = load_result_obj(f'{method_dir}/{file}')
#             K = re.sub('[^0-9]', '', file.split('_')[1])
#             rep = int(file.split('_')[-1][-1])
#             if f'K{K}' not in results[method].keys():
#                 results[method][f'K{K}'] = {}
#             results[method][f'K{K}'][rep] = obj
#     return results

# def make_aa_results_json(analyses: dict, savename: str, A_true=None, Z_true=None):
#     methods = []
#     n_archetypes = []    
#     with_init = []
#     beta_reg = []
#     losses = []
#     NMIs = []
#     MCCs = []
    
#     possible_methods = ['RBOAA', 'OAA', 'CAA'] if 'OSM' not in savename else ['TSAA']
#     for method in possible_methods:
#         for K in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
#             for rep in range(10):
#                 methods.append(method)
#                 n_archetypes.append(K)
#                 if method != 'CAA':
#                     beta_reg.append(True)
#                     with_init.append(True)
#                 else:
#                     beta_reg.append(False)
#                     with_init.append(False)
                
#                 if A_true is not None:
#                     NMIs.append(NMI(analyses[method][f'K{K}'][rep].A), A_true)

#                 if Z_true is not None:
#                     MCCs.append(MCC(analyses[method][f'K{K}'][rep].Z), Z_true)
                
#                 losses.append(list(analyses[method][f'K{K}'][rep].loss))


#     alternating = [False] * len(methods)

#     aa_res_dict = {'method': methods,
#                    'with_init': with_init,
#                    'alternating': alternating,
#                    'beta_reg': beta_reg,
#                    'n_archetypes': n_archetypes,
#                    'loss': losses}
    
#     with open(savename, 'w') as f:
#         json.dump(aa_res_dict, f)

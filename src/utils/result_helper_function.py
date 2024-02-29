import os
from telnetlib import OLD_ENVIRON
from src.utils.eval_measures import calcMI

def result_helper_function(params):
    from AAM import AA
    import numpy as np
    import pandas as pd
    from src.utils.eval_measures import NMI, MCC, BDM
    
    N = 10000
    M = 21
    p = 6
    n_iter = 2000
    reps = 10
    AA_types = ['RBOAA_hotstart', 'RBOAA_alternating', 'RBOAA_betareg', 'RBOAA_alternating_betareg',
                'OAA_hotstart', 'OAA_alternating', 'OAA_betareg', 'OAA_alternating_betareg', 'CAA_']

    #AA_types = ['RBOAA_hotstart', 'OAA_hotstart']

    s = params[0]
    a_param = params[2]
    b_param = params[3]
    sigma_std = params[5]
    synthetic_arch = params[1]
    
    if params[4]:
        analysis_archs = np.arange(2,11)
    else:
        analysis_archs = [synthetic_arch] # set n_archetypes in analysis equal to "true" number

    AA_types_list = []
    analysis_archs_list = []
    reps_list = []
    losses_list = []
    NMIs_list = []
    MCCs_list = []
    BDM_list = []
    sigma_est_list = []

    AAM = AA()
    if b_param == "RB_false":
        AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=False, a_param=a_param, b_param=0,mute=True, sigma_dev=sigma_std)
    else:
        AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=True, a_param=a_param, b_param=b_param,mute=True, sigma_dev=sigma_std)
    
    syn_A = AAM._synthetic_data.A
    syn_Z = AAM._synthetic_data.Z
    syn_betas = AAM._synthetic_data.betas

    for analysis_type in AA_types:
        AA_type = analysis_type.split('_')[0]

        if AA_type == 'CAA':
            lr = 0.1
        else:
            lr = 0.01

        beta_regulators = False
        alternating = False

        if "alternating" in analysis_type:
            alternating = True
        if "betareg" in analysis_type:
            beta_regulators = True

        for analysis_arch in analysis_archs:
            for rep in range(reps):
                
                AA_types_list.append(analysis_type)
                analysis_archs_list.append(analysis_arch)
                reps_list.append(rep)

                AAM.analyse(model_type=AA_type, lr=lr, with_synthetic_data = True, mute=True, K=analysis_arch, n_iter = n_iter, with_hot_start=True, p=p, beta_regulators=beta_regulators, alternating=alternating)

                analysis_A = AAM._synthetic_results[AA_type][0].A
                analysis_Z = AAM._synthetic_results[AA_type][0].Z

                if AA_type == "CAA":
                    loss = AAM._synthetic_results[AA_type][0].loss[-1]
                    BDM_list.append("NaN")
                    sigma_est_list.append("NaN")
                    losses_list.append(loss)
                    NMIs_list.append(NMI(analysis_A, syn_A))
                    MCCs_list.append(MCC(analysis_Z, syn_Z))
                
                else:
                    #import pdb
                    #pdb.set_trace()
                    loss = AAM._synthetic_results[AA_type][0].loss[-1]
                    analysis_betas = AAM._synthetic_results[AA_type][0].b
                    #print(AA_type)
                    #print(analysis_betas.shape)
                    BDM_list.append(BDM(syn_betas,analysis_betas,AA_type))
                    sigma_est_list.append(np.mean(AAM._synthetic_results[AA_type][0].sigma))
                    losses_list.append(loss)
                    NMIs_list.append(NMI(analysis_A, syn_A))
                    MCCs_list.append(MCC(analysis_Z, syn_Z))
                    #print(BDM(syn_betas,analysis_betas,AA_type))
                    #print(NMI(analysis_A,syn_A.T))

    
    CSV_PATH = 'results/DF_ALL_RESULTS.csv'
    
    dataframe = pd.DataFrame.from_dict({
        'sigma': s,
        'sigma_std': sigma_std,
        'synthetic_k': synthetic_arch,
        'a_param': a_param,
        'b_param': b_param,
        'AA_type': AA_types_list, 
        'analysis_k': analysis_archs_list, 
        'rep': reps_list, 
        'loss': losses_list, 
        'NMI': NMIs_list, 
        'MCC': MCCs_list,
        'BDM': BDM_list,
        'Est. sigma': sigma_est_list})
    if not os.path.exists(CSV_PATH):
        df_master = pd.DataFrame(columns = list(dataframe.columns))
        df_master.to_csv(CSV_PATH, index=False)
    df_master = pd.read_csv(CSV_PATH)
    df_master = pd.concat([df_master, dataframe], axis=0, ignore_index=True)
    df_master.to_csv(CSV_PATH, index=False)
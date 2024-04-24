import pandas as pd

def filter_ESS8_data(filepath: str, only_GB: bool = False):
    question_cols = ['SD1', 'PO1', 'UN1', 'AC1', 'SC1', 'ST1', 'CO1', 'UN2', 'TR1', 'HD1', 
                     'SD2', 'BE1', 'AC2', 'SC2', 'ST2', 'CO2', 'PO2', 'BE2', 'UN3', 'TR2', 'HD2']
    
    df_ESS8 = pd.read_csv(filepath, low_memory=False, encoding='utf8') # read data
    
    # filter for GB respondents
    if only_GB:
        df_ESS8 = df_ESS8.loc[df_ESS8['Country'] == 'GB']
    
    # extract questions 
    X = df_ESS8[question_cols].values

    return X.T # array of size n_questions x n_respondents



import numpy as np
import pandas as pd

scp_statements = pd.read_csv("./data/physionet.org/files/ptb-xl/1.0.1/scp_statements.csv", index_col = 0)
ptbxl_database = pd.read_csv("./data/physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv", index_col = 'ecg_id')
print(scp_statements,'\n', ptbxl_database.scp_codes)

for i in ptbxl_database:
    print(i, ptbxl_database[i].isnull().sum())

for i in scp_statements:
    print(i, scp_statements[i].isnull().sum())

mean_age = ptbxl_database['age'].mean()
ptbxl_database.fillna({'age': mean_age}, inplace = True)


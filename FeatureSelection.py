import pandas as pd
from CustomImputer import CustomImputer
from sksurv.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_regression


df = pd.read_csv('FINAL.csv', encoding='latin1')

# Remove irrevalent variables (Dates, Descriptions, State, etc)
df = df.drop(['CAN_DGN_OSTXT', 'REC_DGN_OSTXT', 'DON_CANCER_OTHER_OSTXT', 'PERS_NEXTTX', 'PERS_RELIST',
              'PERS_RETX', 'TFL_ENDTXFU', 'TFL_LAFUDATE', 'CAN_TX_COUNTRY', 'CAN_MAX_AGE', 'CAN_PERM_STATE',
              'CAN_CITIZENSHIP', 'CAN_YEAR_ENTRY_US', 'CAN_EDUCATION', 'CAN_EMPL_STAT', 'CAN_WORK_INCOME',
              'CAN_WORK_NO_STAT', 'CAN_WORK_NO_STAT', 'CAN_ACADEMIC_PROGRESS', 'CAN_ACADEMIC_LEVEL', 'CAN_PRIMARY_PAY',
              'CAN_SECONDARY_PAY','CAN_ENDWLFU', 'DON_CITIZENSHIP', 'REC_PERM_STATE', 'REC_EMPL_STAT_PRE04', 
              'REC_WORK_INCOME', 'REC_WORK_NO_STAT', 'REC_WORK_YES_STAT', 'REC_ACADEMIC_PROGRESS', 'REC_ACADEMIC_LEVEL',
              'REC_PRIMARY_PAY', 'REC_SECONDARY_PAY', 'DON_HOME_STATE','DON_A1', 'DON_A2', 'DON_B1', 'DON_B2', 'DON_DR1', 
              'DON_DR2', 'REC_A1', 'REC_A2', 'REC_B1', 'REC_B2', 'REC_DR1', 'REC_DR2'], axis=1)

# Remove date and ID variables
bad_variables= []
for col in df.columns:
    if col[-3:] == "_DT" or col[-3:] == "_ID":
        if col != "PERS_ID" and col != "DONOR_ID":
            bad_variables.append(col)
            
df = df.drop(bad_variables, axis=1)

# Move Y variable to end
cols = [col for col in df if col != 'DY_TXFL'] + ['DY_TXFL']
df = df[cols]
cols = [col for col in df if col != 'DY_GRFAIL'] + ['DY_GRFAIL']
df = df[cols]

x = df.iloc[:, 1:502]
follow_up = df.iloc[:, 502]
y = df.iloc[:, 503]

# Replace null values with followup data in y variable
y = y.fillna(-1)
survival =[]
for i in range(len(y)):
    if(y[i] == -1):
        survival.append(follow_up[i])
    else:
        survival.append(y[i])
        
# Remove variables with more than 60%  null values
censored_percentage = x.isnull().sum()/x.shape[0]
not_enough_data = []

for col in x.columns:
    if censored_percentage[col] > 0.6:
        not_enough_data.append(col)
        
x = x.drop(not_enough_data, axis=1)

# Impute missing values with mode
x = CustomImputer(strategy='mode').fit_transform(x)

# Removes low-variance categorical features
categorical = x.select_dtypes(['object']).columns
cat = x[categorical]
cat[cat.select_dtypes(['object']).columns] = cat.select_dtypes(['object']).apply(lambda y: y.astype('category'))
cat = OneHotEncoder().fit_transform(cat)
selector = VarianceThreshold(.8 * (1 - .8))
selector.fit_transform(cat)
columns = cat.columns
labels_c = []
for index in selector.get_support(indices=True):
    labels_c.append(columns[index])
selected_categorical = pd.DataFrame(selector.fit_transform(cat), columns=labels_c)

# Feature selection for numeric features
numeric = x.select_dtypes(['float64']).columns
num = x[numeric]
selector = SelectFpr(score_func= f_regression, alpha=0.05)
selected_numeric = selector.fit_transform(num, survival)
columns = num.columns
labels_n = []
for index in selector.get_support(indices=True):
    labels_n.append(columns[index])
selected_numeric = pd.DataFrame(selector.fit_transform(num, survival), columns=labels_n)

# Create final dataset
dataset =  pd.concat([selected_categorical, selected_numeric], axis=1)
dataset['Survival_in_days'] = survival

dataset.to_csv("NEW_DATASET.csv", index=False)

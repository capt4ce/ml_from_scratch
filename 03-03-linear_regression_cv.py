import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_from_scratch.linear_model import LinearRegression
from ml_from_scratch.metrics import mean_squared_error
from ml_from_scratch.model_selection import KFold

# Case 1: finding the best feature
# --------------------------------
# load data
df = pd.read_csv('data/auto.csv')
X = df.drop(columns='mpg')
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_res = X_train.reset_index(drop=True)
y_train_res = y_train.reset_index(drop=True)


cols_list = [["displacement"], ["horsepower"], ["weight"],
             ["displacement", "horsepower"], ["displacement", "weight"],
             ["horsepower", "weight"], 
             ["displacement", "horsepower", "weight"]]
kf = KFold(n_splits=5)

mse_train_list = []
mse_val_list = []
for cols in cols_list:
    mse_train_cols = []
    mse_val_cols = []

    for train_idx, val_idx in kf.split(X_train_res):
        X_train_kf, X_val_kf = X_train_res.loc[train_idx, cols], X_train_res.loc[val_idx, cols]
        y_train_kf, y_val_kf = y_train_res.loc[train_idx], y_train_res.loc[val_idx]

        reg = LinearRegression()
        reg.fit(X_train_kf, y_train_kf)
        y_pred_train = reg.predict(X_train_kf)
        y_pred_val = reg.predict(X_val_kf)

        mse_train = mean_squared_error(y_train_kf, y_pred_train)
        mse_val = mean_squared_error(y_val_kf, y_pred_val)

        mse_train_cols.append(mse_train)
        mse_val_cols.append(mse_val)
    
    mse_train_list.append(np.mean(mse_train_cols))
    mse_val_list.append(np.mean(mse_val_cols))

summary = pd.DataFrame({'cols': cols_list, 'mse_train': mse_train_list, 'mse_val': mse_val_list})

print(summary)
index_best = summary['mse_val'].argmin()
cols_best = summary.loc[index_best, 'cols']
print(f'Best feature: {cols_best}')
print(f"Best valid score: {summary.loc[index_best, 'mse_val']:.2f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_from_scratch.linear_model import LinearRegression

df = pd.read_csv('data/bmd.csv')
X_train = df[['age']]
y_train = df['bmd']

X_test = X_train.copy()+1e-6

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# plotting
fig, ax = plt.subplots()

ax.scatter(X_train, y_train, c='grey', label='observed data')

idx_sort = np.argsort(X_test['age']).to_list()
X_test_sorted = X_test.loc[idx_sort]
y_pred_sorted = y_pred[idx_sort]
ax.plot(X_test_sorted, y_pred_sorted, c='b', label='prediksi')

ax.legend()
ax.grid(linestyle='--')
ax.set_xlabel('age')
ax.set_ylabel('bmd')

plt.show()
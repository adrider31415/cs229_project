print(__doc__)
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


dtrain = '/EXO_mc/TrainingSet.dat'
dtest = '/EXO_mc/TestSet.dat'

# load daset
# x_cols = [cchisqr, chisqcolect, chisq_induct, chisq_collectrestrict, chisq_induct_rest, unshaped_integral, mc_true_energy, current_exo_prediction]

dat = np.loadtxt(dtrain, skiprows = 2)
Xdat = dat[:, :7]
ydat = dat[:, 7]


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=10000)
regr_1.fit(Xdat, ydat)

# Predict
dat_train = np.loadtxt(dtest, skiprows = 2)
Xdat_t = dat_train[:, :7]
ydat_t = dat_train[:, 7]
exo_p = dat_train[:, 8]

y_pred = regr_1.predict(Xdat_t)
rmse = np.sqrt(np.median((y_pred - ydat_t)**2))
rmseexo = np.sqrt(np.median((exo_p - ydat_t)**2))

print rmse
print rmseexo
# Plot the results
plt.plot(ydat_t, y_pred, '.', label = 'BRT')
plt.plot(ydat_t, exo_p, '.', label = 'exo prediction')
plt.xlabel('Deposited charge')
plt.ylabel('Predicted charge')
plt.legend()
plt.show()

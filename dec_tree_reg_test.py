print(__doc__)
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt



# generate daset
# x_cols = [cchisqr, chisqcolect, chisq_induct, chisq_collectrestrict, chisq_induct_rest, unshaped_integral, mc_true_energy, current_exo_prediction]


standard_map = lambda x: np.dot(x, np.arange(len(x))**5)

def gen_data(nsamp, ndat, feature_mapping = standard_map):
    #generates fake data.
    Xdat = np.random.randn(nsamp, ndat)
    ydat = np.sum(Xdat, axis = 1)
    return Xdat, ydat



def fit(Xdat, ydat, md):
    regr_1 = DecisionTreeRegressor(max_depth=md)
    regr_1.fit(Xdat, ydat)
    return regr_1


def mse(reg, Xtest, ytest):
    #rms error per point
    y_pred = regr_1.predict(Xdat_t)
    return np.sqrt(sum((y_pred - ytest)**2/(ytest**2)))/len(ytest)

def mesvsdepth(Xtrain, ytrain, Xtest, ytest, mdmin, mdmax, n):
    mses = np.zeros(n)
    for i, md in enumerate(np.round(np.linspace(mdmin, mdmax, n))):
        reg = fit(Xtrain, ytrain, md)
        mses[i] = mse(reg, Xtest, ytest)
    return np.round(np.linspace(mdmin, mdmax, n)), mses

def plt_error(Xtest, ytest, reg):
    y_pred = reg.predict(Xtest)
    plt.plot(ytest, y_pred, '.')
    plt.xlabel('True value')
    plt.ylabel('predicted value')
    plt.show()

def mse_plot(mds, mses):
    plt.plot(mds, mses, 'o')
    plt.xlabel('Maximum depth')
    plt.ylabel('relative mean squared error per point')
    plt.show()



Xdatt, ydatt = gen_data(1E5, 10)
Xdattest, ydattest = gen_data(1E4, 10)
reg = fit(Xdatt, ydatt, 100)
plt_error(Xdattest, ydattest, reg)

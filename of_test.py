import numpy as np
import optf_util as ofu
import matplotlib.pyplot as plt
import glob
from sklearn.tree import DecisionTreeRegressor

datafs= glob.glob('/EXO_mc/FilterData' + '/WFsShaped*.p')
temp_f = '/EXO_mc/FilterData/templatesShaped.p'

reload = False



if reload:
    dat_objs = ofu.proc_fs(datafs, temp_f)

else:
    dat_objs = ofu.load_fs(datafs)

x = []
y = []
etr = []
for dmobj in dat_objs[:-1]:
    x = np.hstack([x, dmobj.ofamps[:, 0]])
    y = np.hstack([y, dmobj.ofamps[:, 1]])
    etr = np.hstack([etr, dmobj.energies + (dmobj.energies<0)])
    
Xtr = np.vstack([x,y]).T


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=1000)
regr_1.fit(Xtr, etr)

dot = dat_objs[-1]
Xt = np.vstack([dot.ofamps[:, 0], dot.ofamps[:, 1]]).T
et = dot.energies + (dot.energies<0)
epreds = regr_1.predict(Xt)

plt.plot(et, epreds, '.')

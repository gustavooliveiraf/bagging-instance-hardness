import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
import seaborn as sns
from collections import Counter
from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 

from sklearn import tree
from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier

class Main:
    def __init__(self, data):
        self.data = data

    def calc_metrics(self, k_fold, n_times, pool_size, m):
        y = np.array(self.data["CLASS"])
        x = np.array(self.data.drop(axis=1, columns = ["CLASS"]))

        comb = combinations(range(pool_size), 2)
        temp = []
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(x, y):
                X_train, X_test = x[train_index], x[test_index]
                Y_train, Y_test = y[train_index], y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), pool_size)
                BagPercep.fit(x_train, y_train)
                for tupla in comb:
                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(x_test), BagPercep.estimators_[tupla[1]].predict(x_test))
                    temp.append(tupla + (kappa,))
                break
        
        temp.sort(key=lambda tup: tup[2], reverse=True)
        return temp[:m]

    def kNeighborsClassifier(self):
        neigh = KNeighborsClassifier(n_neighbors=7)

        # data_frame = pd.read_csv('./kc2.csv')
        # # enc = LabelEncoder()
        # # df.CLASS = enc.fit_transform(df.CLASS)
        # y = np.array(data_frame["CLASS"])
        # x = np.array(data_frame.drop(axis=1, columns = ["CLASS"]))

        neigh.fit(x, y)

        print(y[0])
        print(y[neigh.kneighbors([x[0]], return_distance=False)[0]])

        print(sum(y[neigh.kneighbors([x[0]], return_distance=False)[0]]!=y[0]))

    def wilcoxon_test(self, x, y):
        p_value = wilcoxon(x, y)
        # print(p_value)

        return p_value

def test(modelo):
    best_classifiers = modelo.calc_metrics(10, 1, 100, 40)
    ensemble = set()
    for i in best_classifiers:
        ensemble.add(i[0])
        ensemble.add(i[1])

    print(list(ensemble))

# test
# df = df.drop(axis=1, columns = ["ID"])
# df = pd.read_csv('./entrada.csv')
# df = pd.read_csv('./cm1.csv')
# df = pd.read_csv('./jm1.csv')
df = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
df.CLASS = enc.fit_transform(df.CLASS)

modelo = Main(df)
# modelo.wilcoxon_test(x,y)

test(modelo)
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
        self.y = np.array(self.data["CLASS"])
        self.x = np.array(self.data.drop(axis=1, columns = ["CLASS"]))
        self.kNeighbors = 7
        self.threshold = 0.5
        self.kdnGreater = [] # ThanThreshold
        self.kdnLess = [] # ThanThreshold

    def calc_metrics(self, k_fold, n_times, pool_size, m):
        comb = combinations(range(pool_size), 2)
        pruning = []
        pruningKdnGreater = []
        pruningKdnLess = []
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                kdnGreater, kdnLess = self.kNeighborsClassifier(x_train, y_train)

                X_validationGreater, X_validationLess = self.x[kdnGreater], self.x[kdnLess]
                Y_validationGreater, Y_validationLess = self.y[kdnGreater], self.y[kdnLess]

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), pool_size)
                BagPercep.fit(x_train, y_train)
                for tupla in comb:
                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(x_train), BagPercep.estimators_[tupla[1]].predict(x_train))
                    pruning.append(tupla + (kappa,))

                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationGreater), BagPercep.estimators_[tupla[1]].predict(X_validationGreater))
                    pruningKdnGreater.append(tupla + (kappa,))

                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationLess), BagPercep.estimators_[tupla[1]].predict(X_validationLess))
                    pruningKdnLess.append(tupla + (kappa,))
                break
        
        pruning.sort(key=lambda tup: tup[2], reverse=True)
        return (pruning[:m], pruningKdnGreater[:m], pruningKdnLess[:m])

    def kNeighborsClassifier(self, x_train, y_train):
        neigh = KNeighborsClassifier(n_neighbors=self.kNeighbors)
        neigh.fit(self.x, self.y)

        kdnGreater = []
        kdnLess = []
        for i in range(len(self.data)):
            kdn = sum(self.y[neigh.kneighbors([x_train[i]], return_distance=False)[0]]!=y_train[i])/self.kNeighbors
            if (kdn > self.threshold):
                kdnGreater.append(i)
            else:
                kdnLess.append(i)
        # print(len(self.kdnGreater), len(self.kdnLess), len(self.kdnLess) + len(self.kdnGreater))
        return (kdnGreater, kdnLess)

    def wilcoxon_test(self, x, y):
        p_value = wilcoxon(x, y)
        # print(p_value)

        return p_value

def test(modelo):
    # modelo.kNeighborsClassifier()
    best_classifiers = modelo.calc_metrics(10, 1, 100, 40) #retorna os 3 conjuntos de classificadores a b c
    ensemble = set()
    ensembleGreater = set()
    ensembleLess = set()
    for i in best_classifiers[0]:
        ensemble.add(i[0])
        ensemble.add(i[1])

    for i in best_classifiers[1]:
        ensembleGreater.add(i[0])
        ensembleGreater.add(i[1])

    for i in best_classifiers[2]:
        ensembleLess.add(i[0])
        ensembleLess.add(i[1])

    print(list(ensemble))

# test
# df = df.drop(axis=1, columns = ["ID"])
# df = pd.read_csv('./entrada.csv')
# df = pd.read_csv('./cm1.csv')
# df = pd.read_csv('./jm1.csv')
data = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
data.CLASS = enc.fit_transform(data.CLASS)

modelo = Main(data)

test(modelo)
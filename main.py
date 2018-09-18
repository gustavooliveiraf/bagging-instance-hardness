from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

neigh = KNeighborsClassifier(n_neighbors=7)

data_frame = pd.read_csv('./kc2.csv')

# enc = LabelEncoder()
# df.CLASS = enc.fit_transform(df.CLASS)

y = np.array(data_frame["CLASS"])
x = np.array(data_frame.drop(axis=1, columns = ["CLASS"]))

neigh.fit(x, y)

print(y[0])
print(y[neigh.kneighbors([x[0]], return_distance=False)[0]])

print(sum(y[neigh.kneighbors([x[0]], return_distance=False)[0]]!=y[0]))

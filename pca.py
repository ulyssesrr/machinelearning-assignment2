#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

classes = [1, 2, 3, 4, 5] 
data_classes = []
data = []

print("Carregando base...")
with open("CNAE-9_reduzido.txt") as f:
	for line in f:
		l = line.split()
		data_classes.append(l[0]);
		data.append(l[1:]);

data = np.array(data).astype(int)
data_classes = np.array(data_classes).astype(int)
print("Base carregada.");

pca = PCA(n_components=2)
X_r = pca.fit(data).transform(data)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, classe in zip("rgbcm", classes, classes):
    plt.scatter(X_r[data_classes == i, 0], X_r[data_classes == i, 1], c=c, label=classe)
plt.legend()
plt.title('PCA CNAE Reduzido (2 Componentes)')


pca = PCA(n_components=2, whiten=True)
X_r = pca.fit(data).transform(data)
plt.figure()
for c, i, classe in zip("rgbcm", classes, classes):
    plt.scatter(X_r[data_classes == i, 0], X_r[data_classes == i, 1], c=c, label=classe)
plt.legend()
plt.title('PCA CNAE Reduzido (2 Componentes)(Branqueado)')

pca = PCA(n_components=1)
X_r = pca.fit(data).transform(data)
plt.figure()
for c, i, classe in zip("rgbcm", classes, classes):
    plt.scatter(X_r[data_classes == i, 0], np.zeros((len(X_r[data_classes == i, 0]),1)), c=c, label=classe)
plt.legend()
plt.title('PCA CNAE Reduzido (1 Componente)')

plt.show()


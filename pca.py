import joblib
import numpy as np
from sklearn.decomposition import PCA


data = joblib.load('./images.joblib')

X = []
labels = []
for label, key in enumerate(data):
    for _ in data[key][0]:
        labels.append(label)
    for x in data[key]:
        X.append(x)


#greyscale images via averaging RGBs
images = np.array(X).mean(axis=3)

#flatten images
images = images.reshape(963, -1)

pca = PCA(n_components=2048)
X_pca = pca.fit_transform(images)

joblib.dump(X_pca, 'pca_representation.joblib')

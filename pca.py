import joblib
import numpy as np
from sklearn.decomposition import PCA


data = joblib.load('./preprocessed_imgs.joblib')
X = np.array(data[0]).reshape(-1, 299, 299, 3)

#greyscale images via averaging RGBs
images = np.array(X).mean(axis=3)

#flatten images
images = images.reshape(963, -1)

pca = PCA(n_components=2048)
X_pca = pca.fit_transform(images)
print(X_pca.shape)
joblib.dump(X_pca, 'pca_representation.joblib')

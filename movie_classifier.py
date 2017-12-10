import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

X = np.array(joblib.load("inception_preprocessed.joblib"))
one_hot, labels, _ = joblib.load("labels.joblib")

#grayscale
X = X.mean(axis=3)

differences = []
for idx, image in enumerate(X[1:]):
    differences.append(image - X[idx])


mean_squared_error = np.array(differences).sum(axis=1).sum(axis=1)**2

# last image doesn't have anything follow it
mean_squared_error = np.insert(mean_squared_error, -1, np.inf)

threshold = 10000000000

# make binary mask for Movie label
y_true = np.array(labels) == 'Movies3of36PhotosEach'

y_predict = mean_squared_error < threshold

accuracy  = accuracy_score(y_true, y_predict)
precision = precision_score(y_true, y_predict)
print(accuracy)
print(precision)

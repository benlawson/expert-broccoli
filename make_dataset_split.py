#standard libraries
import os
import glob

#external libraries
import numpy as np
import joblib
from keras.applications.inception_v3 import InceptionV3 #, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer

X = []
y = []

 # filenames = sorted(glob.glob("/backupdrive/datasets/BUSampleDataSet/*.jpg"))
for folder in sorted(glob.glob("../WinEarthPhotosByKeyword/*")):
    print(folder)
    filenames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for filename in filenames:
        image = img_to_array(load_img(filename, target_size=(299,299)))
        X.append(image)
        y.append(os.path.basename(folder))

inception_model = InceptionV3(include_top=False, pooling='avg')
resnet_model    = ResNet50(include_top=False, pooling='avg')
imgs = X
inception_reps = inception_model.predict(np.array(X))
resnet_reps    = resnet_model.predict(np.array((X)))
encoder = LabelBinarizer()
one_hot_encoded = encoder.fit_transform(y)

joblib.dump((inception_reps), "inception_representations.joblib")
joblib.dump((resnet_reps), "resnet_representations.joblib")
joblib.dump((X), "preprocessed_imgs.joblib")
joblib.dump((one_hot_encoded, y, encoder), "labels.joblib")


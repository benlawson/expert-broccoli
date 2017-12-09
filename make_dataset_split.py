#standard libraries
import os
import glob

#external libraries
import numpy as np
import joblib
from keras.applications.inception_v3 import InceptionV3 # ,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer

X_inception = []
X_resnet = []
y = []

 # filenames = sorted(glob.glob("/backupdrive/datasets/BUSampleDataSet/*.jpg"))
for folder in sorted(glob.glob("WinEarthPhotosByKeyword/*")):
    print(folder)
    filenames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for filename in filenames:
        image_inception = img_to_array(load_img(filename, target_size=(299,299)))
        image_resnet= img_to_array(load_img(filename, target_size=(224,224)))
        X_inception.append(image_inception)
        X_resnet.append(image_resnet)
        y.append(os.path.basename(folder))

inception_model = InceptionV3(include_top=False, pooling='avg')
resnet_model    = ResNet50(include_top=False, pooling='avg')
inception_reps = inception_model.predict(np.array(X_inception))
resnet_reps    = resnet_model.predict(np.array((X_resnet)))
encoder = LabelBinarizer()
one_hot_encoded = encoder.fit_transform(y)

joblib.dump((inception_reps), "inception_representations.joblib")
joblib.dump((resnet_reps), "resnet_representations.joblib")
joblib.dump((X_inception), "inception_preprocessed.joblib")
joblib.dump((X_resnet), "resnet_preprocessed.joblib")
joblib.dump((one_hot_encoded, y, encoder), "labels.joblib")

X_inception = np.array(X_inception)
print(X_inception.shape)

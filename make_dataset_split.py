#standard libraries
import os
import glob

#external libraries
# import numpy as np
import joblib
# from keras.applications.inception_v3 import InceptionV3 # ,preprocess_input
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

exclude_movie = False # SET: True or False
X_inception = []
X_resnet = []
y = []

def get_pca(data):

    images = data.mean(axis=3)

    # flatten images
    images =  images.reshape(-1, 299*299)

    pca = PCA(n_components=2048)
    return pca.fit_transform(images)

print('Exlcuding movie in labels: {}'.format(exclude_movie))
for folder in sorted(glob.glob("./WinEarthPhotosByKeyword/*")): # SPECIFY DIRECTORY
    # exlcude movie filter
    if exclude_movie and 'Movies3of36PhotosEach' in folder:
        continue
    print(folder)
    filenames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for filename in filenames:
        image_inception = img_to_array(load_img(filename, target_size=(299,299)))
        image_resnet= img_to_array(load_img(filename, target_size=(224,224)))
        X_inception.append(image_inception)
        X_resnet.append(image_resnet)
        y.append(os.path.basename(folder))

# initilize models
# inception_model = InceptionV3(include_top=False, pooling='avg')
# resnet_model    = ResNet50(include_top=False, pooling='avg')

# generate different image representation
# inception_reps = inception_model.predict(np.array(X_inception))
# resnet_reps    = resnet_model.predict(np.array(X_resnet))
# pca_reps       = get_pca(np.array(X_inception))

# one hot encoding
encoder = LabelBinarizer()
one_hot_encoded = encoder.fit_transform(y)

# save the representations to disk
path = 'without_movie/' if exclude_movie else 'with_movie/'
os.makedirs(path, exist_ok=True)
# joblib.dump(pca_reps, path + 'pca_representation.joblib')
# joblib.dump((inception_reps), path +"inception_representations.joblib")
# joblib.dump((resnet_reps), path + "resnet_representations.joblib")
joblib.dump((X_inception), path +"inception_preprocessed.joblib")
joblib.dump((X_resnet), path +"resnet_preprocessed.joblib")
joblib.dump((one_hot_encoded, y, encoder), path + "labels.joblib")

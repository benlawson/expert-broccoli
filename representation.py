#standard libraries
import os
import glob
from collections import defaultdict

#external libraries
import numpy as np
import joblib
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# import theano
# theano.config.openmp = True

def preprocess(imgs):
    # shape: (# samples, flattened 1 x 299 x 299 x 3 dimensions)
    preprocessed = []
    for img in imgs:
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        preprocessed.append(img.flatten())
    return preprocessed


model = InceptionV3(include_top=False, pooling='avg')

cat_dict = defaultdict(list)

 # filenames = sorted(glob.glob("/backupdrive/datasets/BUSampleDataSet/*.jpg"))
for folder in sorted(glob.glob("../WinEarthPhotosByKeyword/*")):
    print(folder)
    filenames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for filename in filenames:
        image = img_to_array(load_img(filename, target_size=(299,299)))
        cat_dict[folder].append(image)

#print("loaded all images")
# joblib.dump(cat_dict, "images.joblib")
# cat_dict = joblib.load("images.joblib")
# images = joblib.load("images.joblib")


imgs = []
reps = []
labels = []
for cat in sorted(cat_dict.keys()):
    print("representing {}".format(cat))
    representations = model.predict(np.array(cat_dict[cat]))
    reps.extend(representations)
    preprocessed = preprocess(cat_dict[cat])
    imgs.extend(preprocessed)
    for _ in range(len(cat_dict[cat])):
        labels.append(cat)

joblib.dump((reps, labels), "representations.joblib")
joblib.dump((imgs, labels), "preprocessed_imgs.joblib")


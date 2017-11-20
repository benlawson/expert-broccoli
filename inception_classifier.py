from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import os

def get_images():
    full_paths = []
    img_path = 'dataset/WinEarthPhotosByKeyword/'
    keywords = sorted(os.listdir(img_path))
    for keyword in keywords:
        images = sorted(os.listdir(img_path + keyword))
        for idx, f in enumerate(images):
            full_paths.append(img_path + keyword + '/' + f)
    return full_paths

def preprocess(img_paths):
    X = []
    Y = np.array([])
    # shape: (# samples, flattened 1 x 299 x 299 x 3 dimensions)
    for filename in img_paths:
        img =image.load_img(filename, target_size=(299,299))
        x = np.array(image.img_to_array(img))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X.append(x.flatten())

        y = filename.split('/')[2]
        Y = np.append(Y, y)
    X = np.array(X)
    Y = pd.get_dummies(Y).as_matrix()

    print(X.shape, Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    return X_train, X_test, Y_train, Y_test

'''
Reference: https://keras.io/applications/#usage-examples-for-image-classification-models
'''
process_imgs = True

# X_train shape : (722, 1, 299, 299, 3) y_train shape: (722, 11)
if process_imgs:
    print('Preprocessing data')
    images = get_images()
    joblib.dump(images, 'img_paths.joblib')
    data = preprocess(images)
    joblib.dump(data, 'testdata.joblib')
    X_train, X_test, y_train, y_test = data
else:
    X_train, X_test, y_train, y_test = joblib.load('testdata.joblib')

samples, _ =  X_train.shape
X_train = np.reshape(X_train, (samples, 299, 299, 3))
samples, _ = X_test.shape
X_test= np.reshape(X_test, (samples, 299, 299, 3))

# create base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add fully conected layer
x = Dense(1024, activation='relu')(x)
# add logistic layer to make prediction on 11 classes
predictions = Dense(11, activation='softmax')(x)

# model to train
model = Model(inputs=base_model.input, outputs=predictions)

# train only top layer ie. freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# initial training (only top layer)
model.fit(X_train, y_train, epochs = 10, verbose=1)

# train only the top 2 inception block
# ie. freeze first 249 and unfreeze the rest
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# save (checkpoint) model every 3 epochs
epoch_file = "=finetuned_model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(epoch_file, period=3)
callbacks_list = [checkpoint]

# train to fine-tune convolutional layers
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[categorical_accuracy])
hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1, callbacks=callbacks_list)
print('fine tune scores: {}'.format(hist.history))
model.save('final_inception_model.h5')

scores = model.evaluate(X_test, y_test)
print('Validation score: {} loss {} accuracy'.format(scores[0], scores[1]))

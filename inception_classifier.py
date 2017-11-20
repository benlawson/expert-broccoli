from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import joblib
import os

def get_images():
    full_paths = []
    img_path = 'dataset/WinEarthPhotosByKeyword/'
    keywords = os.listdir(img_path)
    for keyword in keywords:
        images = os.listdir(img_path + keyword)
        for idx, f in enumerate(images):
            full_paths.append(img_path + keyword + '/' + f)
    return full_paths

def preprocess(img_paths):
    X = np.array([])
    Y = np.array([])
    for filename in img_paths:
        print(filename)
        img =image.load_img(filename, target_size=(299,299))
        # 3 dimension
        x = np.array(image.img_to_array(img))
        # 4 dimension
        x = np.expand_dims(x, axis=0)
        # preprocess requires 4 dimensional input
        x = preprocess_input(x)
        X = np.append(X, x.flatten())

        y = filename.split('/')[2]
        Y = np.append(Y, y)
    Y = pd.get_dummies(Y).as_matrix()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    return X_train, X_test, Y_train, Y_test


def main():
    '''
    Reference: https://keras.io/applications/#usage-examples-for-image-classification-models
    '''
    preprocess = False

    # X_train shape : (722, 1, 299, 299, 3) y_train shape: (722, 11)
    if preprocess:
        print('Preprocessing data')
        images = get_images()
        data = preprocess(images)
        joblib.dump(data, 'testdata.joblib')
        X_train, X_test, y_train, y_test = data
    else:
        X_train, X_test, y_train, y_test = joblib.load('testdata.joblib')

    samples, _ =  X_train.shape
    X_train = np.reshape(X_train, (samples, 299, 299, 3))
    samples, _ = X_test.shape
    X_test= np.reshape(X_test, (samples, 299, 299, 3))

    print('creating base model')
    # create base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # adding average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    # logistic layer to make prediction
    predictions = Dense(11, activation='softmax')(x)

    # model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # train only top layer ie. not convolutional layers (not sure what this exactly means...)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    print('Training top layer ')
    # initial training (only top layer)
    model.fit(X_train, y_train, epochs = 10, verbose=1)
    model.save_weights('first_run.h5')

    # to view layer names
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)

    # train top 2 inception block
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # recompile and train to fine-tune convolutional layers
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    print('Fine-tuning using training data')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
    model.save_weights('second_run.h5')

    scores = model.evaluate(X_test, y_test)
    print('Validation score: {} loss {} accuracy'.format(scores[0], [1]))

if __name__ == '__main__':
    main()

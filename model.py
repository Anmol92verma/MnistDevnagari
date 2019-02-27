import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import os
import cv2

NUMBER_OF_PREDICTIONS = 0
IMG_SIZE = 28

DATADIR = "/Users/anmolverma/ml/nhcd/"
CATEGORIES = ["consonants", "numerals", "vowels"]
CSV_FILE = "/Users/anmolverma/ml/labels.csv"
training_data = []
keymap = []


def load_data():
    global NUMBER_OF_PREDICTIONS
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        labels_file = os.path.join(CSV_FILE)
        for subdir in os.listdir(path):
            subdirpath = os.path.join(path, subdir)
            if (os.path.isdir(subdirpath)):
                class_num = NUMBER_OF_PREDICTIONS
                keymap.append([NUMBER_OF_PREDICTIONS, subdirpath])
                NUMBER_OF_PREDICTIONS += 1
                for img in os.listdir(subdirpath):
                    img_array = cv2.imread(os.path.join(subdirpath, img))
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])


load_data()
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3)

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(NUMBER_OF_PREDICTIONS, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)
predictions = model.predict(x_test)
print("predictions " + str(predictions))
print(predictions[0])
print("predictions label " + str(np.argmax(predictions[0])))
print(keymap)

# Convert to TensorFlow Lite model.
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

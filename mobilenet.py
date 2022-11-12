import tensorflow as tf
import numpy as np
import os

import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, GlobalAvgPool2D, Add, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras import Sequential, layers, losses
import cv2 as cv
import matplotlib.image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


data_dir = '/home/praem90/packages/EachOneTeachOne/ImageClassification/Grapevine_Leaves_Image_Dataset/'
test_dir = '/home/praem90/packages/EachOneTeachOne/ImageClassification/Training set Oneplus8'

class_list = os.listdir(data_dir)
class_directory = {"Ak": 0, "Ala_Idris": 1, "Buzgulu": 2, "Dimnit": 3, "Nazli": 4}

train_data = []

train_labels = []

for class_folder in class_list:
    if class_folder[0] != 'G':
        file_names = os.listdir(os.path.join(data_dir, class_folder))
        for file_name in file_names:
            image = matplotlib.image.imread(os.path.join(data_dir, class_folder, file_name))
            try:
                resized = cv.resize(image, (511, 511), interpolation=cv.INTER_AREA)
                train_data.append(resized)
                train_labels.append(class_directory[class_folder])
        #             print(class_directory[class_folder])
            except:
                break
train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_labels = train_labels.reshape((-1, 1))
train_data, train_labels = shuffle(train_data, train_labels)

path_Ak = data_dir + 'Ak'
path_Ala_Idris = data_dir + 'Ala_Idris'
path_Buzgulu = data_dir + 'Buzgulu'
path_Dimnit = data_dir + 'Dimnit'
path_Nazli = data_dir + 'Nazli'

df = pd.DataFrame()


df['images'] = os.listdir(path_Ak) + os.listdir(path_Ala_Idris) + os.listdir(path_Buzgulu) + os.listdir(path_Dimnit) + os.listdir(path_Nazli)


classes = []
paths = []
for image in df['images']:
    class_ = image.split(' (')[0]
    classes.append(class_)
    paths.append(data_dir+class_+'/'+image)

df['classes'] = classes
df['path'] = paths

X_train, X_test, y_train, y_test = train_test_split(df[['path', 'classes']], df[['classes']], test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, shuffle=True, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

mobilenet_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)


# train_generator_mobilenet = mobilenet_datagen.flow_from_directory(
#         data_dir,
#         target_size=(511, 511),
#         batch_size=4,
#         class_mode="categorical",
#         shuffle=True,
# )
# val_generator_mobilenet = mobilenet_datagen.flow_from_directory(
#         data_dir,
#         target_size=(511, 511),
#         batch_size=4,
#         class_mode="categorical",
#         shuffle=True,
# )

train_generator_mobilenet = mobilenet_datagen.flow_from_dataframe(
        X_train,
        x_col='path',
        y_col='classes',
        target_size=(511, 511),
        batch_size=4,
        class_mode="categorical",
        shuffle=True,
)
val_generator_mobilenet = mobilenet_datagen.flow_from_dataframe(
        X_val,
        x_col='path',
        y_col='classes',
        target_size=(511, 511),
        batch_size=4,
        class_mode="categorical",
        shuffle=True,
)

mobilnet_model = tf.keras.applications.MobileNetV2(include_top = False, input_shape = (511,511,3), weights = 'imagenet')

mobilnet_model.trainable = False

inputs = Input((511, 511,3))

base_model_output = mobilnet_model(inputs)

x = GlobalAvgPool2D()(base_model_output)

x = Dense(units= 512, activation='relu')(x)
x = Dense(units= 256, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(units= 5, activation='softmax')(x)


mobilenet = tf.keras.Model(inputs, output)
mobilenet.summary()

mobilenet.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

save_best = tf.keras.callbacks.ModelCheckpoint("best_mobilenet_model_binary_511.h5", save_best_only=True)


history_mobilenet = mobilenet.fit(
      train_generator_mobilenet,
      validation_data=val_generator_mobilenet,
      epochs=10,
      callbacks=[save_best])

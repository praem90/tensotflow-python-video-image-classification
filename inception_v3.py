import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflowjs as tfjs


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf

source_path = "/home/praem90/packages/EachOneTeachOne/ImageClassification/Grapevine_Leaves_Image_Dataset/"
grape_labels = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']

for i in grape_labels:
    print(f"There are {len(os.listdir(os.path.join(source_path, i)))} images of", i)

root_dir = "./grape_dataset"

def create_train_test_dirs(root_path):

    path = os.path.join(root_path, "training")
    os.makedirs(path)
    for i in grape_labels:
        os.makedirs(os.path.join(path,i))
    path = os.path.join(root_path, "testing")
    os.makedirs(path)
    for i in grape_labels:
        os.makedirs(os.path.join(path,i))

create_train_test_dirs(root_dir)


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

  files = []
  for filename in os.listdir(SOURCE):
    file = SOURCE + filename
    if os.path.getsize(file) > 0:
      files.append(filename)
    else:
      print(filename + ' is zero length, so ignoring.')

    import random
    from shutil import copyfile

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

  for filename in training_set:
    src_file = SOURCE + filename
    dest_file = TRAINING + filename
    copyfile(src_file, dest_file)

  for filename in testing_set:
    src_file = SOURCE + filename
    dest_file = TESTING + filename
    copyfile(src_file, dest_file)

training_dir = "./grape_dataset/training"
testing_dir = "./grape_dataset/testing"
split_size = 0.8

for i in grape_labels:
    label_source_dir = source_path + str(i) + "/"
    training_label_dir = os.path.join(training_dir, (str(i)+"/"))
    testing_label_dir = os.path.join(testing_dir, (str(i)+"/"))
    split_data(label_source_dir, training_label_dir, testing_label_dir, split_size)

for i in grape_labels:
    training_label_dir = os.path.join(training_dir, (str(i)+"/"))
    testing_label_dir = os.path.join(testing_dir, (str(i)+"/"))

    print(f"\n\nThere are {len(os.listdir(training_label_dir))} images of", i, "for training")
    print(f"\n\nThere are {len(os.listdir(testing_label_dir))} images of", i, "for testing")


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1.0/255.,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='nearest')

# Pass in the appropriate arguments to the flow_from_directory method
    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=100,
                                                      class_mode='categorical',
                                                      target_size=(300, 300))

# Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    validation_datagen = ImageDataGenerator(rescale=1.0/255.)

# Pass in the appropriate arguments to the flow_from_directory method
    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=50,
                                                                class_mode='categorical',
                                                                target_size=(300, 300))
### END CODE HERE
    return train_generator, validation_generator



train_generator, validation_generator = train_val_generators(training_dir, testing_dir)



from tensorflow.keras import Sequential, layers

model = Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), kernel_initializer="he_uniform", input_shape=(300,300,3), activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), kernel_initializer="he_uniform", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), kernel_initializer="he_uniform", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), kernel_initializer="he_uniform", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=256, kernel_initializer="he_uniform", activation="relu"))
model.add(layers.Dense(units=512, kernel_initializer="he_uniform", activation="relu"))
model.add(layers.Dense(units=64, kernel_initializer="he_uniform", activation="relu"))
model.add(layers.Dense(units=5, kernel_initializer="glorot_uniform", activation="softmax"))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
)


from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


InceptionV3_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))


# The last 15 layers fine tune
for layer in InceptionV3_model.layers[:-15]:
    layer.trainable = False

x = InceptionV3_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(units=512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(units=512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output  = layers.Dense(units=5, activation='softmax')(x)
model = Model(InceptionV3_model.input, output)


model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"])


history = model.fit(train_generator, steps_per_epoch=4, epochs=25, validation_data=validation_generator, validation_steps=2)


model.save('inception_v3')

# tfjs.converters.save_keras_model(model, './js_model/inception_v3_2')

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()



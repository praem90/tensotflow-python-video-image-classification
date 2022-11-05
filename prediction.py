import tensorflow as tf
import numpy as np
import cv2;
from PIL import Image

def writeText(img, class_name):
    img = cv2.resize(img, (500, 500), fx = 0, fy = 0,
                             interpolation = cv2.INTER_CUBIC)
# font
    font = cv2.FONT_HERSHEY_SIMPLEX

    textsize = cv2.getTextSize(class_name, font, 1, 2)[0]

    textX = round((img.shape[1] - textsize[0]) / 2)
    textY = 500 - round((100 - textsize[1]) / 2)

    img = cv2.rectangle( img, (0, 400), (500, 500), (255, 255, 255), -1)

    img = cv2.putText(img, class_name, (textX, textY), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', img)


def get_img_array_from_path(path):
    img = tf.keras.utils.load_img(
        ak_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array


def predict_img():
    ak_path = '/home/praem90/packages/EachOneTeachOne/ImageClassification/Ala idris 05 Nov 4 direction/Ala idris 05 Nov 4 direction/IMG_20221105_105044.jpg'

    img_array = get_img_array_from_path(ak_path)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name = class_names[np.argmax(score)]

    img = cv2.imread(ak_path)

    writeText(img, class_name)



model = tf.keras.models.load_model('./model')

class_names = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']

img_height = 511
img_width = 511

video = cv2.VideoCapture('rtmp://streamspace.live/jptv/livestream')

while True:
    _, frame = video.read()

    #Convert the captured frame into RGB
    img = Image.fromarray(frame, 'RGB')

    #Resizing into dimensions you used while training
    img = img.resize((img_width, img_height))
    img_array = np.array(img)

    #Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name = class_names[np.argmax(score)]

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name = class_names[np.argmax(score)]

    writeText(frame, class_name)

    key=cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


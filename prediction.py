import tensorflow as tf
import numpy as np
import cv2;
from PIL import Image

def writeText(img, class_name):
    img = cv2.resize(img, (round(img.shape[1]/2), round(img.shape[0]/2)), fx = 0, fy = 0,
                             interpolation = cv2.INTER_CUBIC)
# font
    font = cv2.FONT_HERSHEY_SIMPLEX

    textsize = cv2.getTextSize(class_name, font, 1, 2)[0]

    textX = round((img.shape[1] - textsize[0]) / 2)
    textY = textsize[1] * 2

    img = cv2.rectangle( img, (0, 0), (img.shape[1], 100), (255, 255, 255), -1)

    img = cv2.putText(img, class_name, (textX, textY), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', img)


def get_img_array_from_path(path):
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array


def predict_img():
    ak_path = '/home/praem90/packages/EachOneTeachOne/ImageClassification/New training set/IMG_20221105_100342.jpg'
    # ak_path = '/home/praem90/packages/EachOneTeachOne/ImageClassification/Training set Oneplus8/Ala_Idris/IMG_20221104_064741.jpg'
    ak_path = '/home/praem90/Downloads/IMG_20221105_100342.png'

    img_array = get_img_array_from_path(ak_path)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name = class_names[np.argmax(score)]

    img = cv2.imread(ak_path)

    writeText(img, class_name)
    cv2.waitKey(0)

def capture_frame_and_predict(frame):
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

model = tf.keras.models.load_model('inception_v3.h5')

class_names = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']

img_height = 300
img_width = 300

def predict_stream():
    # rtmp_link = 'rtmp://streamspace.live/jptv/livestream'
    rtmp_link = 'rtmp://streamspace.live/eachone/teachone'

    video = cv2.VideoCapture(rtmp_link)

    while True:
        _, frame = video.read()

        if frame is None:
            continue

        capture_frame_and_predict(frame)
        #Convert the captured frame into RGB
        # x = threading.Thread(target=capture_frame_and_predict, args=(frame,))
        # x.start()
        # time.sleep(2000)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()

# predict_img()
predict_stream()

cv2.destroyAllWindows()


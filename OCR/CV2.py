import cv2
import numpy as np
import os, sys
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# Step one get train images for the facial recogintion
# vidcap = cv2.VideoCapture(0)
# for i in range(20):
#     success, image = vidcap.read()
#     cv2.imwrite("Images/test_img_" + str(i) + ".jpg", image)  # save frame as JPEG file
#
# vidcap.release()


def convolve_image(image, kernal):
    img_width, img_height = image.shape[0], image.shape[1]
    x = 0
    all_lists = list()
    while x + kernal[0] < img_width:
        y_list = list()
        y = 0
        while y + kernal[1] < img_height:
            conv_box = np.average(image[x:x+kernal[0], y:y+kernal[1]])
            y_list.append(conv_box)
            y += kernal[1]
        x += kernal[0]
        all_lists.append(y_list)
    return np.array(all_lists)


# Processing the images into grayscale matricies
image_array, label_array = list(), list()
img_dir = 'Images/'
img_files = os.listdir(img_dir)
size = 480, 640
for img_file in img_files:
    # Loading and resizing the image
    resizer = Image.open(img_dir + img_file)
    resized_image = np.array(resizer.resize(size))
    gray = cv2.cvtColor(cv2.imread(img_dir + img_file), cv2.COLOR_BGR2GRAY)
    convolved_image = convolve_image(gray, (5, 5))
    image_array.append(convolved_image)
    if 'test' in img_file:
        label_array.append(1)
    else:
        label_array.append(0)

image_array = np.array(image_array)
label_array = np.array(label_array)

# Set the train test split
X_train, X_test, y_train, y_test = train_test_split(image_array, label_array, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, X_train[0].shape[0], X_train[0].shape[1], 1)
X_test = X_test.reshape(-1, X_test[0].shape[0], X_test[0].shape[1], 1)


# Load the training sets into the model below
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('tanh'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=8, epochs=10, validation_split=0.3)

# Model validation
preds = model.predict(X_test)
preds
y_test

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('resized_2025804.jpg')
img = cv2.resize(img,(96,96))
img = np.reshape(img,[1,96,96,3])
classes = model.predict_classes(img)
print(classes)


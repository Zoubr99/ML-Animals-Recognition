import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

model= tf.keras.models.load_model('my_model3.keras')
img = image.load_img('/Users/zubair/Downloads/', target_size=(150, 150))

x= image.img_to_array(img)
x= np.expand_dims(x, axis=0)

images = np.vstack([x])

classes = model.predict(images, batch_size=10)
print(classes)

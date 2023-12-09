import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def load_image():
    file_path = filedialog.askopenfilename()

    model= tf.keras.models.load_model('my_model.keras')

    img = image.load_img(file_path, target_size=(150, 150))

    x= image.img_to_array(img)
    x= np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)


    print(classes)



    # Define the labels for each class
    labels = ["bull", "butterfly", "cat", "chicken", "dogs"]

    # Find the index of the maximum value in the 'classes' array
    predicted_index = np.argmax(classes)

    # Print the corresponding label
    if predicted_index < len(labels):
        print(f"The predicted class is: {labels[predicted_index]}")
    else:
        print("Unknown class")  # If the predicted index doesn't match any label

root = tk.Tk()

button = tk.Button(root, text='Select Image', command=load_image)
button.pack()


root.mainloop()




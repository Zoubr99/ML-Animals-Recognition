import os
import zipfile
import keras.preprocessing
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
import numpy as np


Training_Directory = 'archive/img/'


#################################    image data generator   ##########################################

Training_data_generator = ImageDataGenerator(rescale=1./255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest')


Training_Data = Training_data_generator.flow_from_directory(
                                                            Training_Directory,
                                                            target_size=(150,150),
                                                            color_mode='rgb',
                                                            class_mode='categorical',
                                                            batch_size=126)

#################################    image data generator   ##########################################


"""

now the data is almost ready to be fed, but first we would need a convultion mdoel

"""

#################################    Creating the Model   ##########################################

""" first create the convultional model and set the maxpooling """

model = tf.keras.models.Sequential([

                tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
    
])

model.summary()

#################################    Creating the Model   ##########################################


"""
now fitting the model and predicting
"""

#################################    fitting and evaluating   ##########################################

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

model.fit(Training_Data, epochs=25,steps_per_epoch=20, verbose=1, validation_split=0.250, validation_steps=3 )

model.save("animals.h5")




#################################    fitting and evaluating   ##########################################
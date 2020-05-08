# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:13:27 2020

@author: deepesh
"""

# part 1 building a cnn

#Importig the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Step 1 : convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64, 64, 3), activation = 'relu'))

#Step 2:Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3 : Flattening
classifier.add(Flatten())

#step 4 : Full connection
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

#part 2 :  Fitting the cnn to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

from keras.models import load_model

classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del classifier  # deletes the existing model

# returns a compiled model
# identical to the previous one
classifier2 = load_model('my_model.h5')


backend.clear_session()
print("The model class indices are:", training_set.class_indices)

#finding the prediction of a image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/prediction/cat.3.jpg', target_size= (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier2.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'dog'
else:
    prediction = 'cat'






















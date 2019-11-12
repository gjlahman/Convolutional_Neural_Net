import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
keras.backend.clear_session()

trainingGenerator = ImageDataGenerator(rescale=1./255)
testGenerator = ImageDataGenerator(rescale=1./255)
validationGenerator = ImageDataGenerator(rescale=1./255)

trainData = trainingGenerator.flow_from_directory("/Users/gjlahman/Documents/GitHub/AI_Final/imgs/train",
                                                  target_size = (200,200),
                                                  batch_size = 16,
                                                  class_mode='binary')

testData = testGenerator.flow_from_directory("/Users/gjlahman/Documents/GitHub/AI_Final/img2/test",
                                                  target_size = (200,200),
                                                  batch_size = 16,
                                                  class_mode='binary')

validationData = validationGenerator.flow_from_directory("/Users/gjlahman/Documents/GitHub/AI_Final/imgs/validation",
                                                  target_size = (200,200),
                                                  batch_size = 16,
                                                  class_mode='binary')


trainingSteps = trainData.n // trainData.batch_size

testSteps = testData.n // testData.batch_size

valSteps = validationData.n // validationData.batch_size

model = Sequential()
model.add(Conv2D(16,(20,20), activation='relu',strides=(4,4), padding='same', input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(25,(5,5), activation='relu',strides=(2,2), padding='same'))
model.add(Conv2D(30,(4,4), activation='relu',strides=(4,4), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='sigmoid'))
model.add(Dropout(.2))
model.add(Dense(84, activation='sigmoid'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid'))

keras.utils.print_summary(model)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=.25, decay=.01),
              metrics=['binary_accuracy'])

history = model.fit_generator(
        trainData,
        steps_per_epoch=trainingSteps,
        epochs=15,
        verbose=1,
        validation_data=validationData,
        validation_steps=valSteps)

model.save("hotdogNetwork.h5")
# Does not work on my computer, keras version too old
#model.evaluate_generator(testData, steps=testSteps, verbose=1)

plt.plot(history.history['binary_accuracy'])

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'] )
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

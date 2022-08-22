import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model
import matplotlib.pyplot as plt

num_features = 64
num_labels = 7
batch_size = 64
epochs = 8
width, height = 72, 72
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

X_train = np.load('./unimodalXtrain.npy')
y_train = np.load('./unimodalytrain.npy')
X_valid = np.load('./unimodalXvalid.npy')
y_valid = np.load('./unimodalyvalid.npy')
X_test = np.load('./unimodalXtest.npy')

files = ['./unimodalytrain.npy', './unimodalyvalid.npy', './unimodalytest.npy']

for file in files:
    print(file)
    y = np.load(file)
    yt = y.tolist()
    y_list = [0,0,0,0,0,0,0]
    for i in range(len(y)):
        yyt = max(yt[i])
        max_index = yt[i].index(yyt)
        y_list[max_index] = y_list[max_index] + 1

    for i in range(7):
        print(str(labels[i]) + ": " + str(y_list[i]))
    print()

# desinging the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))
model.summary()

# compliling the model with adam optimizer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# load weights from partially-trained model
model.load_weights('unimodalModel.h5')

# training the model
history = model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True,
)

print(history.history.keys())
print(history.history['loss'])
print(history.history['val_loss'])

# saving the  model to be used later
fer_json = model.to_json()
with open("unimodalModel-continued.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("unimodalModel-continued.h5")
print("Saved model to disk")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

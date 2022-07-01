import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 19, 7

x = np.load('./fdataX.npy')
y = np.load('./flabels.npy')

x -= np.mean(x, axis = 0)
x /= np.std(x, axis = 0)

X_train = np.load('./modXtrain.npy')
y_train = np.load('./modytrain.npy')
X_valid = np.load('./modXvalid.npy')
y_valid = np.load('./modyvalid.npy')

model = Sequential()

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True)

fer_json = model.to_json()
with open("combinedfer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("combinedfer.h5")
print("Saved model to disk")
model.summary()

''' 
define neural network model (91 stuff)
- fully connected
- activation layer
- fully connected
- softmax --> probabilities --> output
'''

'''
define network model (keras / pytorch/ tensormodel) $
x = face/ hand datapoints $
  y = class $
model.fit(csv) or model.train(csv) (for keras) --> use x and y
'''



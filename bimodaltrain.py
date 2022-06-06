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

x = np.load('./combinedfdataX.npy')
y = np.load('./combinedfLabels.npy')

x -= np.mean(x, axis = 0)
x /= np.std(x, axis = 0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state = 41)

np.save('combinedModXtest', X_test)
np.save('combinedModytest', y_test)

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



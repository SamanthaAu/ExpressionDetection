import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('unimodal.csv')
datapoints = data['pixels'].tolist()
width, height = 72, 72

# getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

# getting labels for training
y = pd.get_dummies(data['emotion']).values

# storing values using numpy
np.save('unimodalDataX', X)
np.save('unimodalLabels', y)

print("Preprocessing Done")
print("Number of features: " + str(len(X[0])))
print("Number of labels: " + str(len(y[0])))
print("Number of examples in dataset: " + str(len(X)))
print("X, y stored in unimodalDataX.npy and unimodalLabels.npy respectively")

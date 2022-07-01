import numpy as np
import matplotlib.pyplot as plt

X_train = np.load('./modXtrain.npy')

for item in X_train:
    plt.imshow(item, cmap="gray")
    plt.show()
    break
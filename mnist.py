from mlxtend.data import loadlocal_mnist
import platform

# Absolute paths to your MNIST data on macOS
train_images = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/Example mnist/train-images-idx3-ubyte"
train_labels = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/Example mnist/train-labels-idx1-ubyte"

# Optional: use different paths if Windows vs. not Windows
if platform.system() == 'Windows':
    # Put Windows paths here if needed
    X, y = loadlocal_mnist(images_path=train_images, labels_path=train_labels)
else:
    # Put macOS paths here
    X, y = loadlocal_mnist(images_path=train_images, labels_path=train_labels)

print('Dimensions:', X.shape, y.shape)
print('First row:', X[0])
print('First label:', y[0])

import numpy as np

print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

np.savetxt(fname='images.csv', 
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='labels.csv', 
           X=y, delimiter=',', fmt='%d')
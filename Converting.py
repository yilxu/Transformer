import os
import struct
import numpy as np
import pandas as pd


###############################################################################
# 1) Utility: Pad or truncate each row to 784 columns => 28x28
###############################################################################
def pad_truncate_to_784(df):
    """
    Given a DataFrame 'df' of shape (N, M), 
    for each row:
      - if M < 784, pad zeros equally front/back,
      - if M > 784, truncate equally front/back,
    so final shape is (N, 784).
    """
    desired_len = 784
    arr_in = df.values  # shape (N, M)
    out_rows = []
    
    for row in arr_in:
        current_len = len(row)
        if current_len < desired_len:
            diff = desired_len - current_len
            front_pad = diff // 2
            back_pad = diff - front_pad
            row_padded = np.pad(row, (front_pad, back_pad),
                                mode='constant', constant_values=0)
        elif current_len > desired_len:
            diff = current_len - desired_len
            front_cut = diff // 2
            back_cut = diff - front_cut
            row_padded = row[front_cut : (current_len - back_cut)]
        else:
            row_padded = row
        
        out_rows.append(row_padded)
    
    out_arr = np.vstack(out_rows)  # shape (N, 784)
    return out_arr


###############################################################################
# 2) Utility: Write images -> standard MNIST 'idx3-ubyte' file
###############################################################################
def write_mnist_images_ubyte(data_2d, output_file):
    """
    data_2d: shape (N, 784), dtype=uint8 in [0..255].
    
    Writes standard MNIST image file with a 16-byte header:
      - magic number (2051) [>i]
      - number of images [>i]
      - rows=28 [>i]
      - cols=28 [>i]
    Followed by N*784 image bytes.
    """
    data_2d = data_2d.astype(np.uint8)
    num_images = data_2d.shape[0]
    
    with open(output_file, "wb") as f:
        # 16-byte header
        f.write(struct.pack(">i", 2051))        # magic
        f.write(struct.pack(">i", num_images))  # N
        f.write(struct.pack(">i", 28))          # rows
        f.write(struct.pack(">i", 28))          # cols
        # image data
        f.write(data_2d.tobytes())


###############################################################################
# 3) Utility: Write labels -> standard MNIST 'idx1-ubyte' file
###############################################################################
def write_mnist_labels_ubyte(labels_1d, output_file):
    """
    labels_1d: shape (N,), dtype=uint8 in [0..9] (typical).
    
    Writes standard MNIST label file with an 8-byte header:
      - magic number (2049) [>i]
      - number of labels [>i]
    Followed by N label bytes.
    """
    labels_1d = labels_1d.astype(np.uint8)
    num_labels = labels_1d.shape[0]
    
    with open(output_file, "wb") as f:
        f.write(struct.pack(">i", 2049))         # magic for labels
        f.write(struct.pack(">i", num_labels))   # N
        f.write(labels_1d.tobytes())


###############################################################################
# 4) Main script: Convert msft_data_255.xlsx to 4 MNIST-like files
###############################################################################
if __name__ == "__main__":
    # A) Read your Excel, which should be in [0..255].
    excel_path = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/msft_data_255.xlsx"
    df = pd.read_excel(excel_path)  # shape (N, M)

    # B) Pad/Truncate each row => shape (N, 784)
    data_784 = pad_truncate_to_784(df)  # shape (N,784), still in [0..255]
    
    # C) Create labels in [0..9].
    #    If you have no real labels, you can create random or all zeros.
    N = data_784.shape[0]
    labels_all = np.random.randint(0, 10, size=N).astype(np.uint8)
    
    # D) Split train=80%, test=20%.
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_idx = int(0.8 * N)
    train_idx = indices[:split_idx]
    test_idx  = indices[split_idx:]
    
    data_train = data_784[train_idx, :]  # shape=(train_count, 784)
    data_test  = data_784[test_idx, :]
    labels_train = labels_all[train_idx]  # shape=(train_count,)
    labels_test  = labels_all[test_idx]
    
    # E) Write them as standard MNIST .ubyte in a chosen folder
    output_dir = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/"  # So your code can do datasets.MNIST("../data", ...)
    os.makedirs(output_dir, exist_ok=True)
    
    train_images_file = os.path.join(output_dir, "train-images-idx3-ubyte")
    train_labels_file = os.path.join(output_dir, "train-labels-idx1-ubyte")
    test_images_file  = os.path.join(output_dir, "t10k-images-idx3-ubyte")
    test_labels_file  = os.path.join(output_dir, "t10k-labels-idx1-ubyte")
    
    # Write images
    write_mnist_images_ubyte(data_train, train_images_file)
    write_mnist_images_ubyte(data_test,  test_images_file)
    
    # Write labels
    write_mnist_labels_ubyte(labels_train, train_labels_file)
    write_mnist_labels_ubyte(labels_test,  test_labels_file)
    
    print(f"Wrote:\n {train_images_file}\n {train_labels_file}\n {test_images_file}\n {test_labels_file}")
    print("Done!")

# Visualize ubyte in csv format 
from mlxtend.data import loadlocal_mnist
import platform

# Absolute paths to your MNIST data on macOS
train_images = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/train-images-idx3-ubyte"
train_labels = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/train-labels-idx1-ubyte"

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

np.savetxt(fname='images3.csv', 
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='labels3.csv', 
           X=y, delimiter=',', fmt='%d')
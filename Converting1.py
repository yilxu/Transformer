#!/usr/bin/env python3

import os
import struct
import numpy as np
import pandas as pd
import platform
from mlxtend.data import loadlocal_mnist

###############################################################################
# 1) Utility: Write images -> standard MNIST 'idx3-ubyte' file
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
        f.write(struct.pack(">i", 2051))        # magic number for images
        f.write(struct.pack(">i", num_images))  # N
        f.write(struct.pack(">i", 28))          # rows
        f.write(struct.pack(">i", 28))          # cols
        # image data
        f.write(data_2d.tobytes())

###############################################################################
# 2) Utility: Write labels -> standard MNIST 'idx1-ubyte' file
###############################################################################
def write_mnist_labels_ubyte(labels_1d, output_file):
    """
    labels_1d: shape (N,), dtype=uint8 in [0..255].
    Writes standard MNIST label file with an 8-byte header:
      - magic number (2049) [>i]
      - number of labels [>i]
    Followed by N label bytes.
    """
    labels_1d = labels_1d.astype(np.uint8)
    num_labels = labels_1d.shape[0]
    
    with open(output_file, "wb") as f:
        # 8-byte header
        f.write(struct.pack(">i", 2049))       # magic number for labels
        f.write(struct.pack(">i", num_labels)) # N
        f.write(labels_1d.tobytes())

###############################################################################
# 3) Utility: Pad/truncate each row to 784 columns => 28x28
###############################################################################
def pad_truncate_to_784(df):
    """
    For each row in df (shape: (N, M)):
      - if M < 784, zero-pad front/back
      - if M > 784, truncate front/back
    Returns an np.array shape (N,784).
    """
    desired_len = 784
    arr_in = df.values  # shape (N,M)
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
    out_arr = np.vstack(out_rows)  # shape (N,784)
    return out_arr


###############################################################################
# 4) Main: read (A) msft_data_255.xlsx for images, (B) return_data_5.xlsx for labels
###############################################################################
if __name__ == "__main__":
    # -- A) Paths --
    base_dir    = "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical"
    msft_file   = os.path.join(base_dir, "msft_data_255.xlsx")   # (N, M) in [0..255]
    return_file = os.path.join(base_dir, "return_data_5.xlsx")      # (N, 1) in [1..5]

    # 1) Read the msft_data_255 => shape (N, M)
    df_msft = pd.read_excel(msft_file)
    #    pad or truncate each row => shape (N,784)
    data_784 = pad_truncate_to_784(df_msft)  # shape (N,784)

    # 2) Read the return_5 => shape (N,1)
    df_return = pd.read_excel(return_file, names=["DailyReturn"], header=0)
    #    flatten => shape (N,)
    label_arr = df_return.values.reshape(-1)

    # 3) Check row counts match
    N_images = data_784.shape[0]
    N_labels = label_arr.shape[0]
    if N_images != N_labels:
        raise ValueError(f"Row mismatch: msft={N_images}, returns={N_labels}")

    N = N_images

    # 4) Do same random shuffle for images & labels
    np.random.seed(1234)  # optional fix seed for reproducibility
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx = int(0.8 * N)
    train_idx = indices[:split_idx]
    test_idx  = indices[split_idx:]

    # 5) Subset for train/test
    train_images = data_784[train_idx, :]
    test_images  = data_784[test_idx, :]
    train_labels = label_arr[train_idx]
    test_labels  = label_arr[test_idx]

    # 6) Write four .ubyte files in the same folder
    train_images_file = os.path.join(base_dir, "train-images-idx3-ubyte")
    train_labels_file = os.path.join(base_dir, "train-labels-idx1-ubyte")
    test_images_file  = os.path.join(base_dir, "t10k-images-idx3-ubyte")
    test_labels_file  = os.path.join(base_dir, "t10k-labels-idx1-ubyte")

    write_mnist_images_ubyte(train_images, train_images_file)
    write_mnist_images_ubyte(test_images,  test_images_file)
    write_mnist_labels_ubyte(train_labels, train_labels_file)
    write_mnist_labels_ubyte(test_labels,  test_labels_file)

    print("Done! 4 files ->")
    print(" Images:", train_images_file, test_images_file)
    print(" Labels:", train_labels_file, test_labels_file)

    # ------------------------------------------------------------------------
    # 7) (Optional) Visualize the train .ubyte in CSV form
    #    We can use mlxtend.data.loadlocal_mnist
    from mlxtend.data import loadlocal_mnist

    trainX, trainY = loadlocal_mnist(
        images_path=train_images_file,
        labels_path=train_labels_file
    )
    print('Train shape:', trainX.shape, trainY.shape)
    print('First row (pixels):', trainX[0])
    print('First label:', trainY[0])
    print('Unique labels:', np.unique(trainY))

    np.savetxt(os.path.join(base_dir, "images3.csv"),
               trainX, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(base_dir, "labels3.csv"),
               trainY, delimiter=',', fmt='%d')

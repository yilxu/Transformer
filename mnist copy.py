import struct
import os
import numpy as np
from mlxtend.data import loadlocal_mnist

def add_mnist_image_header(input_file, output_file, num_items):
    """
    Reads all raw pixel data from 'input_file' (no header),
    then writes a new MNIST-format images file 'output_file'
    with the standard 16-byte header, followed by the raw data.
    """
    # 1) Read the raw data
    with open(input_file, "rb") as f:
        raw_data = f.read()
    
    # 2) Write new file with MNIST 16-byte header
    #    - Magic number for images: 2051 (0x803)
    #    - Number of images (big-endian int)
    #    - Number of rows (28)
    #    - Number of cols (28)
    with open(output_file, "wb") as f:
        # magic number
        f.write(struct.pack('>i', 2051))
        # number of images
        f.write(struct.pack('>i', num_items))
        # rows
        f.write(struct.pack('>i', 28))
        # cols
        f.write(struct.pack('>i', 28))
        # raw pixel bytes
        f.write(raw_data)

def create_dummy_labels_file(file_path, num_items):
    """
    Creates an 8-byte MNIST label header + 'num_items' bytes of label data (all 0).
    """
    with open(file_path, "wb") as f:
        # Magic number 2049 for labels
        f.write(struct.pack('>i', 2049))
        # number of items
        f.write(struct.pack('>i', num_items))
        # all-zero label data
        f.write(b'\x00' * num_items)


if __name__ == "__main__":
    # 1) Your raw images file (no header), each image=784 bytes
    raw_images = "/Users/yilxu/MnistForTransformer/train_images.ubyte"
    
    # 2) Compute how many images
    file_size = os.path.getsize(raw_images)
    num_items = file_size // 784
    print(f"Detected file size: {file_size} => {num_items} images (raw)")

    # 3) Create a new file with MNIST 16-byte header
    images_with_header = "/Users/yilxu/MnistForTransformer/train_images_with_header.ubyte"
    add_mnist_image_header(raw_images, images_with_header, num_items)

    # 4) Create a dummy labels file with 8-byte header
    dummy_labels = "/Users/yilxu/MnistForTransformer/dummy-labels-idx1-ubyte"
    create_dummy_labels_file(dummy_labels, num_items)

    # 5) Now load with mlxtend
    X, y = loadlocal_mnist(images_path=images_with_header, labels_path=dummy_labels)
    print("Dimensions:", X.shape, y.shape)  # e.g. (1991, 784), (1991,)

    # 6) Save pixel data to CSV
    np.savetxt("images2.csv", X, delimiter=",", fmt="%d")
    print("Wrote image data to images2.csv")

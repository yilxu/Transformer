import os
import pandas as pd
import numpy as np
from PIL import Image

# ----------------------------------------------------------------
# 1) Function to convert DataFrame rows -> 28×28 PNG images
# ----------------------------------------------------------------
def rows_to_mnist_images(df, output_folder="MnistForTransformer/0"):
    """
    Convert each row of the DataFrame (assumed in [0,1]) into a
    28×28 grayscale PNG image. Values are scaled to [0,255].
    Rows are padded or truncated to 784 elements.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    data = df.values  # shape: (num_rows, num_cols)
    data_255 = (data * 255).clip(0, 255).astype(np.uint8)
    
    desired_len = 784  # 28 × 28
    
    for idx, row in enumerate(data_255):
        current_len = len(row)
        
        # -- Pad or truncate to length 784 --
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

        # -- Reshape to 28×28 --
        row_2d = row_padded.reshape((28, 28))
        
        # -- Create a grayscale image --
        img = Image.fromarray(row_2d, mode="L")
        
        # -- Save as row_XXXX.png (4-digit zero-padded index) --
        file_index = str(idx + 1).zfill(4)
        file_name = f"row_{file_index}.png"
        img.save(os.path.join(output_folder, file_name))
    
    print(f"Saved {len(data_255)} images to: {output_folder}")


# ----------------------------------------------------------------
# 2) Function to convert all PNG images in a folder to a single .ubyte
# ----------------------------------------------------------------
def convert_images_to_ubyte(image_folder, output_file):
    """
    Convert all PNG images in 'image_folder' to a single UBYTE file,
    stacking each flattened grayscale image as rows in a NumPy array.
    """
    image_data = []
    
    # Read each .png in sorted filename order
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).convert("L")  # Ensure grayscale
            arr = np.array(img, dtype=np.uint8).flatten()
            image_data.append(arr)
    
    # Stack all images vertically -> shape: (num_images, 784)
    image_data = np.vstack(image_data)
    
    # Write to .ubyte
    with open(output_file, "wb") as f:
        f.write(image_data.tobytes())
    
    print(f"Images successfully converted to UBYTE format -> {output_file}")


# ----------------------------------------------------------------
# 3) Function to split images into train/test .ubyte files (80/20)
# ----------------------------------------------------------------
def convert_images_to_ubyte_split(image_folder, train_file, test_file, train_ratio=0.8):
    """
    Reads all PNG images in 'image_folder' (sorted by filename),
    flattens them to 784 bytes each, and splits them into
    train and test sets. Saves train set in 'train_file',
    and test set in 'test_file'.
    """
    image_data = []
    
    # 1) Gather all flattened images
    png_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".png")]
    for filename in png_files:
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path).convert("L")  # grayscale
        arr = np.array(img, dtype=np.uint8).flatten()  # shape = (784,)
        image_data.append(arr)
    
    image_data = np.vstack(image_data)  # shape = (num_images, 784)
    num_images = image_data.shape[0]
    
    # 2) Shuffle them (for a random train/test split)
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    image_data = image_data[indices, :]  # reorder rows
    
    # 3) Split into train & test
    split_index = int(train_ratio * num_images)  # 80% by default
    train_data = image_data[:split_index, :]
    test_data  = image_data[split_index:, :]
    
    # 4) Write train data to .ubyte
    with open(train_file, "wb") as f:
        f.write(train_data.tobytes())
    
    # 5) Write test data to .ubyte
    with open(test_file, "wb") as f:
        f.write(test_data.tobytes())
    
    print(f"Split {num_images} images into: {train_data.shape[0]} train, {test_data.shape[0]} test")
    print(f"Train => {train_file}")
    print(f"Test  => {test_file}")


# ----------------------------------------------------------------
# 4) Main script
# ----------------------------------------------------------------
if __name__ == "__main__":
    # -- Step A: Read the Excel file --
    excel_path = (
        "/Users/yilxu/Desktop/Document/PhD/Research/"
        "Hemang Subramanian/Technical/msft_data_normalized_min_max.xlsx"
    )
    df = pd.read_excel(excel_path)
    
    # -- Step B: Create folder "MnistForTransformer/0" for images --
    image_folder = "MnistForTransformer/0"
    os.makedirs(image_folder, exist_ok=True)
    
    # -- Step C: Convert each row to 28×28 PNGs in the folder --
    rows_to_mnist_images(df, output_folder=image_folder)
    
    # -- Step D1: (Optional) Convert all PNG images to a single .ubyte file --
    ubyte_file_path = "images.ubyte"
    convert_images_to_ubyte(image_folder, ubyte_file_path)
    
    # -- Step D2: Split into train and test .ubyte files, 80/20 --
    train_ubyte_file = "train_images.ubyte"
    test_ubyte_file  = "test_images.ubyte"
    convert_images_to_ubyte_split(
        image_folder=image_folder,
        train_file=train_ubyte_file,
        test_file=test_ubyte_file,
        train_ratio=0.8
    )

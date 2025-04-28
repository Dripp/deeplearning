import os
import shutil

def split_images(source_dir, dest_dir1, dest_dir2, dest_dir3):
    """
    Splits images from source_dir into three destination directories,
    preserving the 0.pgm to 4999.pgm naming convention.

    Args:
        source_dir (str): Path to the directory containing the images.
        dest_dir1 (str): Path to the first destination directory (0.pgm - 3999.pgm).
        dest_dir2 (str): Path to the second destination directory (4000.pgm - 4499.pgm).
        dest_dir3 (str): Path to the third destination directory (4500.pgm - 4999.pgm).
    """

    # Create destination directories if they don't exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    os.makedirs(dest_dir3, exist_ok=True)

    # Move images to destination directories
    for i in range(5001):
        source_file = os.path.join(source_dir, str(i) + ".pgm")
        if not os.path.exists(source_file):
            print(f"Warning: Source file {source_file} not found. Skipping.")
            continue # Skip to the next file if it's missing

        if i <= 4000:
            dest_dir = dest_dir1
        elif i <= 4500:
            dest_dir = dest_dir2
        else:
            dest_dir = dest_dir3

        dest_file = os.path.join(dest_dir, str(i) + ".pgm")
        shutil.copy2(source_file, dest_file)  # Copy the file, preserving metadata

    print("Images split and copied successfully.")

# Set the paths
source_directory = r"D:\python_project\yenet\data\stego"
destination_directory_1 = r"D:\python_project\yenet\data\train\stego_train"
destination_directory_2 = r"D:\python_project\yenet\data\test\stego_test"
destination_directory_3 = r"D:\python_project\yenet\data\val\stego_val"

# Call the function
split_images(source_directory, destination_directory_1, destination_directory_2, destination_directory_3)
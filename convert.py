import os
import json
import shutil
import random
import numpy as np


labels = []


def convert_json_to_txt(json_file, txt_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    if len(data['annotations']) == 0:
        print(f"WARNING: {json_file} contains no annotations.")
        return None

    with open(txt_file, 'w') as file:
        for annotation in data['annotations']:
            # Extract relevant information
            x, y, z = annotation['3dbbox.location']
            dx, dy, dz = annotation['3dbbox.dimension']
            heading_angle = annotation['3dbbox.rotation_y']
            # category_name = "Vehicle" if annotation['3dbbox.category'] == "car" else "Pedestrian" # Modify as needed for other categories
            # category_name = annotation['3dbbox.category'].replace(" ", "")
            category_name = annotation['3dbbox.category']

            # Write formatted data to TXT file
            file.write(f"{x}, {y}, {z}, {dx}, {dy}, {dz}, {heading_angle}, {category_name}\n")

            return f'{x}, {y}, {z}, {dx}, {dy}, {dz}, {heading_angle}, {category_name}'


def search_directory(root_path, labels_path, lidar_copy_path):
    # Create the labels and lidar_copy directories if they don't exist
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    if not os.path.exists(lidar_copy_path):
        os.makedirs(lidar_copy_path)

    for root, dirs, files in os.walk(root_path):
        # Check if 'lidar/roof/' is in the current path
        if 'lidar' in dirs:
            lidar_path = os.path.join(root, 'lidar')
            if 'roof' in os.listdir(lidar_path):
                roof_path = os.path.join(lidar_path, 'roof')
                print(f"Found 'lidar/roof/' directory at: {roof_path}")

                # Print all JSON files in the 'lidar/roof/' directory
                for file in os.listdir(roof_path):
                    if file.endswith('.json'):
                        print(f"JSON file: {file}")

                        # Convert JSON file to TXT file
                        json_file = os.path.join(roof_path, file)
                        txt_file = os.path.join(labels_path, file[:-5] + '.txt')
                        if convert_json_to_txt(json_file, txt_file) is not None:
                            labels.append(file[:-5])

                    if file.endswith('.bin'):
                        print(f"BIN file: {file}")

                        # Copy BIN file to lidar_copy_path
                        bin_file = os.path.join(roof_path, file)
                        lidar_file = os.path.join(lidar_copy_path, file[:-4] + '.npy')

                        print(f"Copying {bin_file} to {lidar_file}")
                        # shutil.copy2(bin_file, lidar_file)
                        # np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4)
                        np.save(lidar_file, np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4))

base_path = '/home/ubuntu/Desktop/woojin/OpenPCDet/data/robust/'


# Start the search from a specified root directory, e.g., '/'
search_directory(base_path, labels_path=f'{base_path}labels', lidar_copy_path=f'{base_path}points')

# Split labels into train and val
def split_labels(labels, train_ratio=0.8):
    # Randomly shuffle the list
    random.shuffle(labels)

    # Calculate the number of labels for training
    train_size = int(len(labels) * train_ratio)

    # Split the labels list
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]

    return train_labels, val_labels


# Split the labels into training and validation sets
train_labels, val_labels = split_labels(labels)

# Optionally, save these labels to files
def save_labels_to_file(labels, file_path):
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(label + '\n')

# Create the ImageSets directory if it doesn't exist
if not os.path.exists(f'{base_path}ImageSets'):
    os.makedirs(f'{base_path}ImageSets')

# Save train and validation labels to separate files
save_labels_to_file(train_labels, f'{base_path}ImageSets/train.txt')
save_labels_to_file(val_labels, f'{base_path}ImageSets/val.txt')

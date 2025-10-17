import os
import shutil
import json
from tqdm import tqdm

def harmonize_classifier_data():
    """
    Reads the raw CNR-EXT-Patches-150x150 dataset and organizes it
    into a structured format required by image classifier frameworks like PyTorch.
    """
    print("Starting data harmonization for classifier models...")

    # --- 1. Define Paths ---
    raw_data_root = os.path.join('data', 'raw', 'CNR-EXT-Patches-150x150')
    processed_data_root = os.path.join('data', 'processed', 'classifier')

    if not os.path.exists(raw_data_root):
        print(f"Error: Raw data not found at '{raw_data_root}'.")
        print("Please ensure the 'CNR-EXT-Patches-150x150' folder is in 'data/raw/'.")
        return

    # --- 2. Create Output Directories ---
    print(f"Creating output directory structure at '{processed_data_root}'...")
    os.makedirs(os.path.join(processed_data_root, 'train', 'occupied'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_root, 'train', 'vacant'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_root, 'test', 'occupied'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_root, 'test', 'vacant'), exist_ok=True)

    # --- 3. Process the Label Files ---
    sets_to_process = [
        ('train.txt', 'train'),
        ('test.txt', 'test')
    ]

    label_map = {'0': 'vacant', '1': 'occupied'}

    for label_file_name, set_name in sets_to_process:
        label_file_path = os.path.join(raw_data_root, 'LABELS', label_file_name)

        print(f"\nProcessing '{label_file_name}'...")

        try:
            with open(label_file_path, 'r') as f:
                lines = f.readlines()

                for line in tqdm(lines, desc=f"Copying '{set_name}' images"):
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue 
                    
                    relative_image_path, label = parts
                    
                    # --- THIS IS THE CORRECTED LINE ---
                    # We now explicitly include 'PATCHES' in the path construction.
                    source_image_path = os.path.join(raw_data_root, 'PATCHES', relative_image_path.replace('/', os.sep))
                    
                    class_name = label_map.get(label)
                    
                    if class_name and os.path.exists(source_image_path):
                        destination_folder = os.path.join(processed_data_root, set_name, class_name)
                        destination_image_path = os.path.join(destination_folder, os.path.basename(relative_image_path))
                        
                        shutil.copy(source_image_path, destination_image_path)

        except FileNotFoundError:
            print(f"  Warning: Label file not found at '{label_file_path}'. Skipping.")

    print("\nData harmonization for classifiers complete!")
    print(f"Processed data is now available in '{processed_data_root}'.")

def harmonize_detector_data():
    """
    Processes the 'archive' (COCO JSON format) dataset for object detector models like YOLO.
    It reads the _annotations.coco.json file, converts labels to YOLO .txt format,
    and organizes the image and label files into a new structure.
    """
    print("\n--- Starting Harmonization for Detector Models ---")

    # --- 1. Define Paths ---
    raw_data_root = os.path.join('data', 'raw', 'archive')
    processed_data_root = os.path.join('data', 'processed', 'detector')

    if not os.path.exists(raw_data_root):
        print(f"Error: Raw data not found at '{raw_data_root}'.")
        print("Please ensure the 'archive' folder is in 'data/raw/'.")
        return

    # --- 2. Create Output Directories ---
    print(f"Creating output directory structure at '{processed_data_root}'...")
    for split in ['train', 'test', 'valid']:
        os.makedirs(os.path.join(processed_data_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(processed_data_root, 'labels', split), exist_ok=True)

    # --- 3. Process each split (train, test, valid) ---
    for split in ['train', 'test', 'valid']:
        source_dir = os.path.join(raw_data_root, split)
        annotation_file = os.path.join(source_dir, '_annotations.coco.json')

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found for '{split}' set at '{annotation_file}'. Skipping.")
            continue

        print(f"\nProcessing '{split}' set for detectors...")

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # --- LEARNING POINT: Create a lookup map ---
        # Create a dictionary to map image IDs to their file names and dimensions.
        images_map = {image['id']: (image['file_name'], image['width'], image['height']) for image in coco_data['images']}
        
        # Create a dictionary to hold the labels for each image.
        # The key will be the image filename, the value will be a list of its YOLO labels.
        labels_by_image = {fname: [] for fname, _, _ in images_map.values()}

        # Loop through all annotations in the JSON file.
        for ann in tqdm(coco_data['annotations'], desc=f"Converting '{split}' annotations"):
            image_id = ann['image_id']
            category_id = ann['category_id'] # In this dataset, it will likely always be the same.
            bbox = ann['bbox'] # This is [x_min, y_min, width, height] in COCO format.

            if image_id not in images_map:
                continue

            # We will use class '0' for 'occupied'.
            class_id = 0 

            file_name, img_width, img_height = images_map[image_id]
            
            # --- LEARNING POINT: Coordinate Conversion & Normalization ---
            x_min, y_min, box_width, box_height = bbox
            
            # 1. Convert COCO's [x_min, y_min, width, height] to YOLO's [x_center, y_center, width, height].
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            # 2. Normalize all values by dividing by the image's dimensions.
            norm_x_center = x_center / img_width
            norm_y_center = y_center / img_height
            norm_width = box_width / img_width
            norm_height = box_height / img_height

            # Append the formatted label string to the list for this image.
            labels_by_image[file_name].append(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}")

        # --- 4. Write Files and Copy Images ---
        print(f"Writing label files and copying images for '{split}' set...")
        for image_filename, yolo_labels in tqdm(labels_by_image.items(), desc=f"Saving '{split}' files"):
            # Write the .txt label file
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(processed_data_root, 'labels', split, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_labels))

            # Copy the corresponding image file
            source_image_path = os.path.join(source_dir, image_filename)
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, os.path.join(processed_data_root, 'images', split, image_filename))

    print("\n✓ Detector data harmonization is complete!")
    print(f"Processed data is now available in '{processed_data_root}'.")


if __name__ == '__main__':
    # Comment out the function you are not currently working on.
    # harmonize_classifier_data() 
    # harmonize_detector_data()
    print("\nAll data processing tasks finished!")



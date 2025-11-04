# This script processes a COCO JSON annotation file to create a dataset for classifiers.
# suitable for training or testing a classifier by cropping bounding box regions from images.
# I used miniconda for managing dependencies and executing this script.

import os
import cv2
import json
import tqdm
import sys

def create_classifier_dataset(json_path, image_dir, output_dir):
    """
    Reads a COCO JSON annotation file and crops the bounding boxes
    from the images, saving them into class-specific folders.
    """
    print(f"Loading annotation file from: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ ERROR: Annotation file not found at '{json_path}'")
        print("   Please make sure you have created 'davao_annotations.json' using the converter.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Create a mapping of category IDs to class names
    if 'categories' not in data:
        print("❌ ERROR: 'categories' key not found in JSON file.")
        return
        
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    print(f"Found class names: {list(category_map.values())}")

    # 2. Create output directories for each class
    for class_name in category_map.values():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    print(f"Output directories created in: {output_dir}")

    # 3. Create a mapping of image IDs to image file paths
    if 'images' not in data:
        print("❌ ERROR: 'images' key not found in JSON file.")
        return
        
    image_map = {img['id']: os.path.join(image_dir, img['file_name']) for img in data['images']}
    
    if 'annotations' not in data:
        print("❌ ERROR: 'annotations' key not found in JSON file.")
        return

    # 4. Loop through annotations, crop, and save
    print(f"\nProcessing {len(data['annotations'])} annotations...")
    count_saved = 0
    
    for ann in tqdm.tqdm(data['annotations'], desc="Cropping spots"):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # Format is [x, y, width, height]
        
        # Get class name and image path
        class_name = category_map.get(category_id)
        image_path = image_map.get(image_id)
        
        if not class_name or not image_path:
            print(f"Warning: Skipping annotation {ann['id']} (missing class or image info).")
            continue
            
        if not os.path.exists(image_path):
            print(f"Warning: Skipping annotation {ann['id']} (image file not found: {image_path}).")
            continue

        # Load the full image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Skipping annotation {ann['id']} (could not read image: {image_path}).")
            continue
            
        # Get bounding box coordinates
        x, y, w, h = [int(v) for v in bbox]
        
        # Crop the region of interest
        cropped_img = img[y : y + h, x : x + w]
        
        # Check if crop is valid
        if cropped_img.size == 0:
            print(f"Warning: Skipping annotation {ann['id']} (zero size crop). Bbox: {bbox}")
            continue
            
        # Create a unique filename for the cropped image
        # Use annotation_id to ensure uniqueness
        crop_filename = f"crop_img{image_id}_ann{ann['id']}.jpg"
        save_path = os.path.join(output_dir, class_name, crop_filename)
        
        # Save the cropped image
        cv2.imwrite(save_path, cropped_img)
        count_saved += 1

    print(f"\n✅ Done! Successfully saved {count_saved} cropped images to '{output_dir}'.")

if __name__ == "__main__":
    # Check for TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ ERROR: TensorFlow is not installed.")
        print("   Please install it by running: python -m pip install tensorflow")
        sys.exit(1)
        
    # Check for tqdm
    try:
        import tqdm
    except ImportError:
        print("❌ ERROR: tqdm is not installed.")
        print("   Please install it by running: python -m pip install tqdm")
        sys.exit(1)

    # Define paths
    davao_json_path = os.path.join("data", "real_world_test_set", "davao_annotations.json")
    davao_image_dir = os.path.join("data", "real_world_test_set")
    classifier_output_dir = os.path.join("data", "processed", "davao_classifier_test_set")
    
    # Run the function
    create_classifier_dataset(davao_json_path, davao_image_dir, classifier_output_dir)
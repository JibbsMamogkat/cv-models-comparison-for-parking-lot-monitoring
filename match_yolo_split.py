import os
import json
import tqdm
import sys

def match_yolo_split_to_coco(master_json_path, yolo_train_img_dir, yolo_valid_img_dir, output_dir):
    """
    Reads a master COCO JSON file and two folders (train/valid) from a
    YOLO split, and creates new COCO JSON files that match that split.
    """
    print("--- Matching YOLO Split to COCO JSON ---")
    
    # --- 1. Load the Master "Answer Key" JSON ---
    print(f"Loading master JSON from: {master_json_path}")
    if not os.path.exists(master_json_path):
        print(f"❌ ERROR: Master annotation file not found at '{master_json_path}'")
        return
        
    with open(master_json_path, 'r') as f:
        master_data = json.load(f)
        
    master_images = master_data['images']
    master_annotations = master_data['annotations']
    categories = master_data['categories']

    # --- 2. Create a lookup map from filename -> image_object ---
    # This map lets us find an image's ID and details just by its name
    filename_to_image_map = {img['file_name']: img for img in master_images}

    # --- 3. Read the filenames from the YOLO split folders ---
    try:
        train_filenames = {f for f in os.listdir(yolo_train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
        valid_filenames = {f for f in os.listdir(yolo_valid_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find YOLO split directories.")
        print(f"   Missing: {e.filename}")
        print("   Please ensure your groupmate has run their script and the folders exist.")
        return

    print(f"Found {len(train_filenames)} image files in YOLO train dir.")
    print(f"Found {len(valid_filenames)} image files in YOLO valid dir.")

    # --- 4. Create new lists of images and IDs based on the split ---
    new_train_images = []
    new_valid_images = []
    train_image_ids = set()
    valid_image_ids = set()

    for filename in train_filenames:
        if filename in filename_to_image_map:
            img_obj = filename_to_image_map[filename]
            new_train_images.append(img_obj)
            train_image_ids.add(img_obj['id'])
        
    for filename in valid_filenames:
        if filename in filename_to_image_map:
            img_obj = filename_to_image_map[filename]
            new_valid_images.append(img_obj)
            valid_image_ids.add(img_obj['id'])

    # --- 5. Filter the master annotations using the new ID sets ---
    new_train_annotations = [ann for ann in master_annotations if ann['image_id'] in train_image_ids]
    new_valid_annotations = [ann for ann in master_annotations if ann['image_id'] in valid_image_ids]

    print(f"Matched {len(new_train_images)} train images and {len(new_train_annotations)} annotations.")
    print(f"Matched {len(new_valid_images)} valid images and {len(new_valid_annotations)} annotations.")

    # --- 6. Create the new JSON structures ---
    train_json = {
        "info": master_data.get('info', {}),
        "licenses": master_data.get('licenses', []),
        "images": new_train_images,
        "annotations": new_train_annotations,
        "categories": categories
    }
    valid_json = {
        "info": master_data.get('info', {}),
        "licenses": master_data.get('licenses', []),
        "images": new_valid_images,
        "annotations": new_valid_annotations,
        "categories": categories
    }

    # --- 7. Save the new, matched JSON files ---
    os.makedirs(output_dir, exist_ok=True)
    train_output_path = os.path.join(output_dir, "davao_finetune_train_MATCHED.json")
    valid_output_path = os.path.join(output_dir, "davao_finetune_valid_MATCHED.json")
    
    try:
        print(f"Saving matched training JSON to: {train_output_path}")
        with open(train_output_path, 'w') as f:
            json.dump(train_json, f, indent=4)
            
        print(f"Saving matched validation JSON to: {valid_output_path}")
        with open(valid_output_path, 'w') as f:
            json.dump(valid_json, f, indent=4)
            
        print("\n✅ Done. JSON files matching the YOLO split have been created.")
    except Exception as e:
        print(f"❌ ERROR saving output JSON files: {e}")


if __name__ == "__main__":
    # Check for tqdm
    try:
        import tqdm
    except ImportError:
        print("❌ ERROR: tqdm is not installed.")
        print("   Please install it by running: python -m pip install tqdm")
        sys.exit(1)

    # --- 1. Path to your MASTER Davao annotations ---
    MASTER_JSON_PATH = os.path.join("data", "real_world_test_set", "davao_annotations.json")
    
    # --- 2. Paths to your groupmate's YOLO split folders ---
    #    (This assumes their script `split_davao_dataset.py` saved to this location)
    YOLO_TRAIN_IMG_DIR = os.path.join("data", "processed", "davao_finetune_set_YOLO", "images", "train")
    YOLO_VALID_IMG_DIR = os.path.join("data", "processed", "davao_finetune_set_YOLO", "images", "valid")
    
    # --- 3. Where to save your NEW JSON files ---
    OUTPUT_JSON_DIR = os.path.join("data", "processed", "davao_finetune_set_JSON_MATCHED")

    # --- Run the conversion ---
    print("This script will create COCO JSON files that match your groupmate's YOLO split.")
    print(f"Reading master JSON from: {MASTER_JSON_PATH}")
    print(f"Reading YOLO train images from: {YOLO_TRAIN_IMG_DIR}")
    print(f"Reading YOLO valid images from: {YOLO_VALID_IMG_DIR}")
    print(f"Saving new JSONs to: {OUTPUT_JSON_DIR}")
    
    input("\nPress Enter to start...")

    match_yolo_split_to_coco(MASTER_JSON_PATH, YOLO_TRAIN_IMG_DIR, YOLO_VALID_IMG_DIR, OUTPUT_JSON_DIR)
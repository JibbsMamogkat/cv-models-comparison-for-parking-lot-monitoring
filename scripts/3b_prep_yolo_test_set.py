import json
import os
from tqdm import tqdm

def convert_coco_to_yolo():
    """
    Loads the Davao COCO JSON annotations and converts them into 
    YOLOv8 compatible .txt files (one per image).
    
    It will:
    1. Read 'davao_annotations.json'.
    2. Ignore 'space-empty' (id 1).
    3. Convert 'space-occupied' (id 2) to class '0'.
    4. Normalize coordinates.
    5. Write a .txt file for each image into a new 'labels' folder.
    """
    print("--- Converting COCO JSON to YOLO .txt format ---")

    # --- 1. Define Paths ---
    test_set_root = os.path.join('data', 'real_world_test_set', 'davao_detector_test_set')
    json_path = os.path.join(test_set_root, 'annotations', 'davao_annotations.json')
    
    # This is our new output folder
    output_labels_dir = os.path.join(test_set_root, 'labels')
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # --- 2. Load the Original JSON ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Original annotation file not found at: {json_path}")
        return
    
    print(f"Loaded original annotations from: {json_path}")

    # --- 3. Create Mappings ---
    # Create a map of {image_id: (filename, width, height)}
    images_map = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}
    
    # Create a map to hold {filename: [list of yolo strings]}
    yolo_labels = {img['file_name']: [] for img in data['images']}

    # --- 4. Convert Annotations ---
    ignored_count = 0
    converted_count = 0

    for ann in tqdm(data['annotations'], desc="Converting annotations"):
        if ann['category_id'] == 1: # 'space-empty'
            ignored_count += 1
            continue
        
        if ann['category_id'] == 2: # 'space-occupied'
            image_id = ann['image_id']
            if image_id not in images_map:
                continue

            file_name, img_width, img_height = images_map[image_id]
            
            # COCO bbox is [x_min, y_min, width, height]
            x_min, y_min, box_width, box_height = ann['bbox']

            # Convert to YOLO: [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            norm_x_center = x_center / img_width
            norm_y_center = y_center / img_height
            norm_width = box_width / img_width
            norm_height = box_height / img_height
            
            # Our YOLO model only knows one class: id 0
            class_id = 0 
            
            yolo_string = f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}"
            yolo_labels[file_name].append(yolo_string)
            converted_count += 1

    print(f"Conversion complete:")
    print(f"  > Converted {converted_count} 'space-occupied' annotations.")
    print(f"  > Ignored {ignored_count} 'space-empty' annotations.")

    # --- 5. Write all the .txt files ---
    print(f"Writing {len(yolo_labels)} label files to '{output_labels_dir}'...")
    for file_name, labels in yolo_labels.items():
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_filepath = os.path.join(output_labels_dir, txt_filename)
        
        with open(txt_filepath, 'w') as f:
            f.write("\n".join(labels))

    print(f"\n✓ Successfully created YOLO .txt label files.")

if __name__ == '__main__':
    convert_coco_to_yolo()

# This script converts VOC XML annotations to COCO JSON format
# specifically for the Davao Real World Test Set.
# I used miniconda for managing dependencies and executing this script.

import os
import xml.etree.ElementTree as ET
import json
import cv2
import tqdm

def convert_voc_to_coco(xml_dir, image_dir, output_json_path):
    """
    Converts a directory of VOC XML annotations into a single COCO JSON file
    using a FIXED category map to match the original training data.
    """
    print(f"Starting conversion from VOC XML to COCO JSON...")
    print(f"XML source: {xml_dir}")
    print(f"Image source: {image_dir}")

    # --- 1. Initialize COCO JSON structure ---
    coco_output = {
        "info": {
            "description": "Davao Real World Test Set",
            "year": 2025,
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [] 
    }

   
    category_map = {
        "space-empty": 1,
        "space-occupied": 2
    }

    coco_output["categories"].append({
        "id": 1,
        "name": "space-empty",
        "supercategory": "spaces"
    })
    coco_output["categories"].append({
        "id": 2,
        "name": "space-occupied",
        "supercategory": "spaces"
    })
    
    print(f"Using fixed category map: {category_map}")
    
    annotation_id_counter = 1 # Annotation IDs must start from 1 because 0 is not used in the fast-rcnn model

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"❌ ERROR: No .xml files found in '{xml_dir}'.")
        return

    print(f"Found {len(xml_files)} XML annotation files.")

    # --- 2. Loop through all XML files ---
    for image_id, xml_file in enumerate(tqdm.tqdm(xml_files, desc="Converting XML files"), 1):
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_node = root.find('filename')
            if filename_node is None:
                print(f"Warning: Skipping {xml_file} (no <filename> tag).")
                continue
            
            image_filename = filename_node.text
            image_path = os.path.join(image_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Skipping {xml_file} (image file not found: {image_path}).")
                continue

            size_node = root.find('size')
            if size_node is None:
                try:
                    img = cv2.imread(image_path)
                    height, width, _ = img.shape
                except Exception as e:
                    print(f"Warning: Skipping {xml_file} (could not read image size: {e}).")
                    continue
            else:
                width = int(size_node.find('width').text)
                height = int(size_node.find('height').text)

            image_info = {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)

            # --- 3. Loop through all "object" tags in the XML ---
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                category_id = category_map.get(class_name)
                
                # Check if the label from Makesense.ai is valid
                if category_id is None:
                    print(f"\n❌ ERROR: Unknown class name '{class_name}' in {xml_file}.")
                    print(f"   Please use one of the valid names: {list(category_map.keys())}")
                    continue # Skip this annotation

                # Get bounding box [xmin, ymin, xmax, ymax]
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                annotation_info = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id, # Use the CORRECT ID (1 or 2)
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                    "segmentation": [] 
                }
                coco_output["annotations"].append(annotation_info)
                annotation_id_counter += 1
                
        except Exception as e:
            print(f"❌ ERROR processing file {xml_file}: {e}")

    # --- 4. Save the final COCO JSON file ---
    print("\nConversion complete. Saving COCO JSON file...")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"✅ Successfully saved to: {output_json_path}")
        print("\n--- Final Category Map Used ---")
        print(coco_output["categories"])
        print("----------------------------")
    except Exception as e:
        print(f"❌ ERROR saving JSON file: {e}")

if __name__ == "__main__":
    # --- DEFINE PATHS HERE ---
    
    # 1. The folder where 50 Davao images are.
    DAVAO_IMAGE_DIR = os.path.join("data", "real_world_test_set")
    
    # 2. The folder where unzipped the XML files from Makesense.ai.
    XML_SOURCE_DIR = os.path.join("data", "real_world_test_set", "xml_annotations")
    
    # 3. The name of the final JSON file to create.
    JSON_OUTPUT_PATH = os.path.join(DAVAO_IMAGE_DIR, "davao_annotations.json")
    
    # --- End of paths ---

    print("If created from Makesense.ai: Please make sure you have done the following:")
    print(f"1. Placed all your 50 Davao images in: '{DAVAO_IMAGE_DIR}'")
    print(f"2. Placed all 50 exported .xml files in: '{XML_SOURCE_DIR}'")
    print(f"3. Labels in Makesense.ai MUST be 'space-empty' or 'space-occupied'. following the EXACT spelling and casing from trained dataset.")
    print(f"4. The output file will be saved as: '{JSON_OUTPUT_PATH}'")
    input("\nPress Enter to start the conversion...")
    
    if not os.path.exists(XML_SOURCE_DIR):
        print(f"❌ ERROR: The XML source directory does not exist: '{XML_SOURCE_DIR}'")
        print("Please create it and move your .xml files there.")
    elif not os.path.exists(DAVAO_IMAGE_DIR):
        print(f"❌ ERROR: The image directory does not exist: '{DAVAO_IMAGE_DIR}'")
    else:
        convert_voc_to_coco(XML_SOURCE_DIR, DAVAO_IMAGE_DIR, JSON_OUTPUT_PATH)
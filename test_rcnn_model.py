# This Script is used to test a trained Faster R-CNN model for parking spot detection.
# It allows predicting on a single image or evaluating on test datasets, providing flexibility for different use cases.
# I used miniconda for managing dependencies and executing this script.


# --- Standard Python Libraries ---
import os
import cv2
import random 
import torch
import numpy as np
import argparse 

# --- Detectron2 Libraries ---
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances, load_coco_json 
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def setup_cfg(config_file, weights_file, confidence_threshold=0.7):
    """
    Loads configuration from training and sets up the predictor or evaluation config.
    """
    cfg = get_cfg()
    
    base_config_path = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(base_config_path)
    
    try:
        cfg.merge_from_file(config_file)
        print(f"Loaded training configuration from {config_file}")
        # --- VERIFICATION STEP ---
        print(f"VERIFICATION: NUM_CLASSES loaded from config.yaml is: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    except FileNotFoundError:
        print(f"Warning: Training config file not found at {config_file}. Using defaults.")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # Fallback, must match training
        
    cfg.MODEL.WEIGHTS = weights_file 
    print(f"Attempting to load model weights from: {cfg.MODEL.WEIGHTS}")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold 
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    cfg.freeze()
    return cfg

def predict_single_image(cfg, metadata):
    """
    Prompts for an image path, displays predictions, and saves the output image.
    Uses custom coloring (Red/Green) and counts.
    """
    image_path_input = input("\nPlease enter the full path to your image file: ")
    image_path = image_path_input.strip().strip('"').strip("'") 

    if not os.path.exists(image_path):
        print(f"❌ ERROR: Input image file not found at '{image_path}'")
        return

    print(f"\n--- Running Prediction on: {os.path.basename(image_path)} ---")
    
    try:
        predictor = DefaultPredictor(cfg) 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load the model with the specified weights.")
        print(f"   Error details: {e}")
        return
        
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ ERROR: Could not read image file at '{image_path}'. Is it a valid image?")
        return
        
    print("Running inference...")
    outputs = predictor(img_bgr)
    
    # --- Custom Visualization Logic ---
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    
    class_names = metadata.thing_classes
    
    # Define colors (BGR format for OpenCV)
    color_occupied = (0, 0, 255) # Red
    color_empty = (0, 255, 0)   # Green
    color_other = (255, 0, 0)   # Blue
    
    count_occupied = 0
    count_empty = 0
    
    output_image_draw = img_bgr.copy()
    
    if boxes is not None and scores is not None and classes is not None:
        print(f"Found {len(boxes)} instances (before confidence threshold).")
        for i in range(len(boxes)):
            score = scores[i]
            if score < cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST: 
                continue
                
            class_id = classes[i].item()
            box = boxes[i].tensor.numpy().astype(int)[0] 
            
            # --- FIX for class_id out of range ---
            if class_id >= len(class_names):
                print(f"Warning: Predicted class ID {class_id} is out of range for metadata (len={len(class_names)}). Skipping.")
                continue

            class_name = class_names[class_id]
            color = color_other 
            
            # --- Check class names for coloring ---
            if class_name == "space-occupied": 
                color = color_occupied
                count_occupied += 1
            elif class_name == "space-empty": 
                color = color_empty
                count_empty += 1
            # --- End Check ---

            cv2.rectangle(output_image_draw, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{class_name}: {score:.0%}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_origin = (box[0], box[1] - baseline - 2)
            if text_origin[1] < text_height:
                 text_origin = (box[0], box[1] + text_height + 2)
            cv2.rectangle(output_image_draw, text_origin, (text_origin[0] + text_width, text_origin[1] - text_height - baseline), color, -1)
            cv2.putText(output_image_draw, label, (text_origin[0], text_origin[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Detected spots (Conf > {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.0%}): Occupied={count_occupied}, Empty={count_empty}")

    # --- Save the output image ---
    output_save_dir = "./prediction_outputs/custom_predictions"
    os.makedirs(output_save_dir, exist_ok=True) 
    
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_predicted_color_conf{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}{ext}" 
    output_save_path = os.path.join(output_save_dir, output_filename)
    
    try:
        cv2.imwrite(output_save_path, output_image_draw)
        print(f"✅ Prediction image saved to: '{output_save_path}'")
    except Exception as e:
        print(f"❌ ERROR: Could not save prediction image. Error: {e}")

    print(f"Displaying prediction (Confidence threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}). Press any key in the image window to close.")
    cv2.imshow(f"Prediction Result (Conf: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f})", output_image_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--- Single Image Prediction Complete ---")

def evaluate_test_set(cfg, dataset_name, metadata): 
    """
    Runs evaluation, prints results, AND saves ONE random prediction image from the test set.
    """
    print(f"\n--- Running Evaluation & Saving ONE Prediction for Dataset: {dataset_name} ---")
    
    # --- Part 1: Standard Evaluation (Calculate Scores) ---
    try:
        model_for_eval = DefaultPredictor(cfg).model 
        print("Model loaded successfully for evaluation scoring.")
        
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR, tasks=("bbox",))
        

        print("Forcing evaluator to use the model's metadata (3 classes)...")
        evaluator._metadata = metadata 
        
        test_loader_for_eval = build_detection_test_loader(cfg, dataset_name)
        
        print("Starting official evaluation scoring (this may take a few minutes)...")
        results = inference_on_dataset(model_for_eval, test_loader_for_eval, evaluator)
        
        print("\n--- Evaluation Results (Scores) ---")
        print(results)
        print("--- Score Calculation Complete ---")
        
    except Exception as e:
        print(f"❌ ERROR during evaluation scoring: {e}")
        import traceback
        traceback.print_exc()
        return 

    # --- Part 2: Save ONE Random Visualization ---
    print("\n--- Visualizing and Saving ONE Random Test Set Prediction ---")
    output_viz_dir = f"./prediction_outputs/{dataset_name}_example" 
    os.makedirs(output_viz_dir, exist_ok=True)
    
    try:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if not dataset_dicts: 
             print(f"❌ ERROR: Dataset dictionary for '{dataset_name}' is empty. Cannot visualize.")
             return
             
        random_image_dict = random.choice(dataset_dicts) 
        file_path = random_image_dict["file_name"]
        
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Randomly selected image file not found at {file_path}. Cannot visualize.")
            return
            
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ ERROR: Could not read image {file_path}. Cannot visualize.")
            return

        predictor_for_viz = DefaultPredictor(cfg) 
        outputs = predictor_for_viz(img)
        
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = out.get_image()[:, :, ::-1]
        
        base_filename = os.path.basename(file_path)
        name, ext = os.path.splitext(base_filename)
        output_filename = f"{name}_random_predicted_conf{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}{ext}"
        output_save_path = os.path.join(output_viz_dir, output_filename)
        
        try:
            cv2.imwrite(output_save_path, output_image)
            print(f"✅ Random prediction image saved to: '{output_save_path}'")
        except Exception as e:
            print(f"❌ ERROR saving random prediction image: {e}")
            
        print(f"Displaying random prediction for {base_filename}. Press any key to close.")
        cv2.imshow(f"Random Test Prediction (Conf: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f})", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"❌ ERROR during visualization saving: {e}")
            
    print("\n--- Full Evaluation Complete ---")


def main(args): 
    """
    Main function to handle user choice and run prediction or evaluation.
    """
    setup_logger() 
    
    # --- Define Paths ---
    output_dir = "./output_local" 
    config_path = os.path.join(output_dir, "config.yaml") 
    weights_path_rcnn = os.path.join(output_dir, "fast_rcnn.pth") 
    weights_path_final = os.path.join(output_dir, "model_final.pth")
    

    weights_path = None
    if os.path.exists(weights_path_rcnn):
        weights_path = weights_path_rcnn
        print(f"Found renamed weights file: '{weights_path}'")
    elif os.path.exists(weights_path_final):
        weights_path = weights_path_final
        print(f"Found default weights file: '{weights_path}'")
    else:
         print(f"❌ ERROR: Trained model weights not found at '{weights_path_rcnn}' or '{weights_path_final}'")
         return

    if not os.path.exists(config_path):
         print(f"❌ ERROR: Configuration file not found at '{config_path}'")
         print("   Please run the training script (train_local_rcnn.py) for a moment to generate it.")
         return
         
    # --- Register dataset(s) required for metadata and evaluation ---
    # Standard Test Set
    base_data_path_std = os.path.join("data", "raw", "archive") 
    std_test_dataset_name = "parking_test_eval" 
    metadata_std = None
    
    # Davao Test Set
    base_data_path_davao = os.path.join("data", "real_world_test_set")
    davao_test_dataset_name = "davao_test_eval"
    davao_json_path = os.path.join(base_data_path_davao, "davao_annotations.json")
    metadata_davao = None

    # Register Standard Test Set
    try:
        if std_test_dataset_name not in DatasetCatalog.list():
            register_coco_instances(std_test_dataset_name, {}, 
                                    os.path.join(base_data_path_std, "test", "_annotations.coco.json"), 
                                    os.path.join(base_data_path_std, "test"))
        metadata_std = MetadataCatalog.get(std_test_dataset_name)
        print("Standard test dataset metadata loaded/registered.")
    except Exception as e:
        print(f"Warning: Could not register standard test set. Option 2 may fail. Error: {e}")

    # Register Davao Test Set
    try:
        if davao_test_dataset_name not in DatasetCatalog.list():
            if not os.path.exists(davao_json_path):
                 print(f"Warning: Davao annotation file not found at '{davao_json_path}'. Cannot register Davao test set (Option 3).")
            else:
                 register_coco_instances(davao_test_dataset_name, {}, 
                                         davao_json_path, 
                                         base_data_path_davao) 
                 metadata_davao = MetadataCatalog.get(davao_test_dataset_name)
                 print("Davao real-world test dataset metadata loaded/registered.")

                 if metadata_std is not None:
                     print("Applying metadata from standard test set to Davao test set to ensure class ID consistency.")
                     davao_metadata_obj = MetadataCatalog.get(davao_test_dataset_name)
                     davao_metadata_obj.thing_classes = metadata_std.thing_classes
                     davao_metadata_obj.thing_dataset_id_to_contiguous_id = metadata_std.thing_dataset_id_to_contiguous_id
                     davao_metadata_obj.class_aliases = metadata_std.class_aliases 
                 else:
                     print("Warning: Standard metadata not loaded. Davao evaluation might fail due to class ID mismatch.")
                     davao_metadata_obj = MetadataCatalog.get(davao_test_dataset_name)
                     davao_metadata_obj.thing_classes = ["spaces", "space-empty", "space-occupied"]
                     davao_metadata_obj.thing_dataset_id_to_contiguous_id = {0: 0, 1: 1, 2: 2}

    except Exception as e:
        print(f"Warning: Could not register Davao test set. Option 3 may fail. Error: {e}")

    # --- Ask User for Action ---
    print("\nWhat would you like to do?")
    print("  1: Predict on a single custom image (and save the result).")
    print("  2: Run evaluation and save ONE random image from the STANDARD test dataset.")
    print("  3: Run evaluation and save ONE random image from the DAVAO test dataset.") 
    
    choice = input("Enter your choice (1, 2, or 3): ")

    # --- Execute Chosen Action ---
    conf_thresh = args.confidence if args.confidence is not None else 0.7 
    
    if choice == '1':
        cfg_predict = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh)
        pred_metadata = metadata_std 
        if pred_metadata is None: 
             pred_metadata = MetadataCatalog.get(davao_test_dataset_name) 
        if pred_metadata is None or not hasattr(pred_metadata, 'thing_classes'): 
             pred_metadata = MetadataCatalog.get("__dummy")
             if not hasattr(pred_metadata, 'thing_classes') or not pred_metadata.thing_classes:
                 pred_metadata.thing_classes = ["spaces", "space-empty", "space-occupied"] 
             print("Using dummy metadata for visualization.")
        predict_single_image(cfg_predict, pred_metadata)
        
    elif choice == '2':
        cfg_eval_std = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh) 
        if std_test_dataset_name in DatasetCatalog.list():
             if metadata_std is None: metadata_std = MetadataCatalog.get(std_test_dataset_name)
             evaluate_test_set(cfg_eval_std, std_test_dataset_name, metadata_std) 
        else:
             print("❌ ERROR: Standard test dataset was not registered successfully. Cannot run evaluation.")
             
    elif choice == '3': 
        cfg_eval_davao = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh) 
        if davao_test_dataset_name in DatasetCatalog.list():

             if metadata_std is not None:
                 print("Using standard metadata map for Davao evaluation.")
                 evaluate_test_set(cfg_eval_davao, davao_test_dataset_name, metadata_std) # Pass metadata_std
             else:
                 print("❌ ERROR: Standard test set metadata not loaded. Cannot run Davao evaluation correctly.")
        else:
             print(f"❌ ERROR: Davao test dataset ('{davao_test_dataset_name}') was not registered successfully.")
             print(f"   Ensure '{davao_json_path}' exists and images are in '{base_data_path_davao}'.")

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict or evaluate using a trained Faster R-CNN model.")
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=None, 
        help="Confidence threshold for prediction visualization (e.g., 0.5). Default=0.7"
    )
    args = parser.parse_args()

    main(args)


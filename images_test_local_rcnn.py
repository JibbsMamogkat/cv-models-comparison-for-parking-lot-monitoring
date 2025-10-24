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
        print(f"VERIFICATION: NUM_CLASSES loaded from config.yaml is: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    except FileNotFoundError:
        print(f"Warning: Training config file not found at {config_file}. Using defaults.")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
        
    cfg.MODEL.WEIGHTS = weights_file 
    print(f"Attempting to load model weights from: {cfg.MODEL.WEIGHTS}")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold 
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    cfg.freeze()
    return cfg

def predict_single_image(cfg, metadata):
    """
    Prompts for an image path, displays predictions, and saves the output image.
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
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ERROR: Could not read image file at '{image_path}'. Is it a valid image?")
        return
        
    print("Running inference...")
    outputs = predictor(img)
    
    # --- Visualize ---
    v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = out.get_image()[:, :, ::-1] 
    
    # --- Save the output image ---
    output_save_dir = "./prediction_outputs/single_predictions" 
    os.makedirs(output_save_dir, exist_ok=True) 
    
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_predicted_conf{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}{ext}" 
    output_save_path = os.path.join(output_save_dir, output_filename)
    
    try:
        cv2.imwrite(output_save_path, output_image)
        print(f"✅ Prediction image saved to: '{output_save_path}'")
    except Exception as e:
        print(f"❌ ERROR: Could not save prediction image. Error: {e}")

    print(f"Displaying prediction (Confidence threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}). Press any key in the image window to close.")
    cv2.imshow(f"Prediction Result (Conf: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f})", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--- Single Image Prediction Complete ---")
    
def evaluate_test_set(cfg, dataset_name, base_data_path, metadata): 
    """
    Runs evaluation, prints results, AND saves ONE random prediction image from the test set.
    """
    print(f"\n--- Running Evaluation & Saving ONE Prediction for Dataset: {dataset_name} ---")
    
    # --- Part 1: Standard Evaluation (Calculate Scores) ---
    try:
        model_for_eval = DefaultPredictor(cfg).model 
        print("Model loaded successfully for evaluation scoring.")
        
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        test_loader_for_eval = build_detection_test_loader(cfg, dataset_name)
        
        print("Starting official evaluation scoring (this may take a few minutes)...")
        results = inference_on_dataset(model_for_eval, test_loader_for_eval, evaluator)
        
        print("\n--- Evaluation Results (Scores) ---")
        print(results)
        print("--- Score Calculation Complete ---")
        
    except Exception as e:
        print(f"❌ ERROR during evaluation scoring: {e}")
        # Stop if scoring fails, as visualization depends on a working model setup
        return 

    # --- Part 2: Save ONE Random Visualization ---
    print("\n--- Visualizing and Saving ONE Random Test Set Prediction ---")
    # Define where to save the image
    output_viz_dir = "./prediction_outputs/test_set_single_example"
    os.makedirs(output_viz_dir, exist_ok=True)
    
    try:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if not dataset_dicts: # Check if list is empty
             print(f"❌ ERROR: Dataset dictionary for '{dataset_name}' is empty. Cannot visualize.")
             return
             
        random_image_dict = random.choice(dataset_dicts) # Pick a random image entry
        file_path = random_image_dict["file_name"]
        
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Randomly selected image file not found at {file_path}. Cannot visualize.")
            return
            
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ ERROR: Could not read image {file_path}. Cannot visualize.")
            return

        # Create predictor for visualization
        predictor_for_viz = DefaultPredictor(cfg)
        outputs = predictor_for_viz(img)
        
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = out.get_image()[:, :, ::-1]
        
        # Create output filename
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
    
    # --- Determine which weights file to use ---
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
         return
         
    base_data_path = os.path.join("data", "raw", "archive") 
    test_dataset_name = "parking_test_eval" 
    metadata = None
    registration_needed = test_dataset_name not in DatasetCatalog.list()

    try:
        if registration_needed:
            register_coco_instances(test_dataset_name, {}, 
                                    os.path.join(base_data_path, "test", "_annotations.coco.json"), 
                                    os.path.join(base_data_path, "test"))
        metadata = MetadataCatalog.get(test_dataset_name)
        print("Dataset metadata loaded/registered for class names and evaluation.")
    except Exception as e:
        print(f"Warning: Could not register dataset. Evaluation scoring might fail. Error: {e}")
        metadata = MetadataCatalog.get("__dummy")
        if not hasattr(metadata, 'thing_classes'):
            metadata.thing_classes = [f"class_{i}" for i in range(3)] 
        print("Using dummy metadata for visualization.")

    # --- Ask User for Action ---
    print("\nWhat would you like to do?")
    print("  1: Predict on a single custom image (and save the result).")
    print("  2: Run evaluation and save ONE random prediction image from the standard test dataset.") # Updated text
    
    choice = input("Enter your choice (1 or 2): ")

    # --- Execute Chosen Action ---
    conf_thresh = args.confidence if args.confidence is not None else 0.7 
    
    if choice == '1':
        cfg_predict = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh)
        predict_single_image(cfg_predict, metadata)
    elif choice == '2':
        cfg_eval = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh) 
        if test_dataset_name in DatasetCatalog.list():
             evaluate_test_set(cfg_eval, test_dataset_name, base_data_path, metadata) 
        else:
             print("❌ ERROR: Test dataset was not registered successfully. Cannot run evaluation.")
    else:
        print("Invalid choice. Please enter 1 or 2.")


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


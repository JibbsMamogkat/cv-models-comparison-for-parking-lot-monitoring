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


def setup_cfg(config_file, weights_file, confidence_threshold=0.7, is_eval=False):
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
    
    if is_eval:
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        print("--- FIX APPLIED: Setting DATALOADER.NUM_WORKERS = 0 for stable evaluation. ---")
        
    cfg.freeze()
    return cfg

def create_image_grid(image_list, grid_rows, grid_cols, img_height, img_width):
    """
    Takes a list of images and stitches them into a grid.
    Pads with blank images if the list is shorter than the grid size.
    """
    num_images = len(image_list)
    num_needed = grid_rows * grid_cols
    
    padded_images = list(image_list)
    while len(padded_images) < num_needed:
        print("Warning: Not enough images for full grid. Adding blank image.")
        blank_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Image", (img_width // 4, img_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        padded_images.append(blank_img)
        
    rows_list = []
    for i in range(grid_rows):
        start_index = i * grid_cols
        end_index = start_index + grid_cols
        img_row = padded_images[start_index:end_index]
        
    
        resized_row = [cv2.resize(img, (img_width, img_height)) for img in img_row]
        
        h_stacked_row = np.hstack(resized_row)
        rows_list.append(h_stacked_row)
    
    final_grid_image = np.vstack(rows_list)
    return final_grid_image

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

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    
    class_names = metadata.thing_classes
    
    color_occupied = (0, 0, 255) 
    color_empty = (0, 255, 0)   
    color_other = (255, 0, 0)  
    
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
            
            if class_id >= len(class_names):
                print(f"Warning: Predicted class ID {class_id} is out of range for metadata (len={len(class_names)}). Skipping.")
                continue

            class_name = class_names[class_id]
            color = color_other 
            
            if class_name == "space-occupied": 
                color = color_occupied
                count_occupied += 1
            elif class_name == "space-empty": 
                color = color_empty
                count_empty += 1


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
def evaluate_and_save_grids(cfg, dataset_name, metadata):
    """
    Runs evaluation, prints results, AND saves 4x4 GRIDS for
    both Ground Truth and Predictions from the test set.
    """
    print(f"\n--- Running Evaluation & Saving 4x4 Grids for: {dataset_name} ---")

    # --- Part 1: Standard Evaluation (Calculate Scores) ---
    try:
        model_for_eval = DefaultPredictor(cfg).model
        print("Model loaded successfully for evaluation scoring.")
        
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR, tasks=("bbox",))
        print("Forcing evaluator to use the model's metadata...")
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

    # --- Part 2: Save 4x4 Prediction & Ground Truth Grids ---
    print("\n--- Visualizing and Saving 4x4 Grids (16 images) ---")
    output_viz_dir = f"./prediction_outputs/{dataset_name}_4x4_grid"
    os.makedirs(output_viz_dir, exist_ok=True)
    
    GRID_ROWS = 4
    GRID_COLS = 4
    IMG_WIDTH = 640 
    IMG_HEIGHT = 640
    NUM_IMAGES = GRID_ROWS * GRID_COLS

    try:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if not dataset_dicts:
            print(f"❌ ERROR: Dataset dictionary for '{dataset_name}' is empty. Cannot visualize.")
            return
            
        images_to_process = dataset_dicts[:NUM_IMAGES]
        if len(images_to_process) < NUM_IMAGES:
            print(f"Warning: Dataset only has {len(images_to_process)} images. Grid will be padded.")

        predictor = DefaultPredictor(cfg)
        
        processed_gt_images = []
        processed_pred_images = []

        print(f"Processing {len(images_to_process)} images for grids...")
        
        for d in images_to_process:
            file_path = d["file_name"]
            if not os.path.exists(file_path):
                print(f"⚠️ Skipping missing image: {file_path}")
                continue
                
            img = cv2.imread(file_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {file_path}")
                continue

            # --- Create Ground Truth Image ---
            v_gt = Visualizer(img[:, :, ::-1], metadata, scale=1.0)
            out_gt = v_gt.draw_dataset_dict(d)
            img_gt = out_gt.get_image()[:, :, ::-1]
            processed_gt_images.append(img_gt) # Resize later in helper

            # --- Create Prediction Image ---
            outputs = predictor(img)
            v_pred = Visualizer(img[:, :, ::-1], metadata, scale=1.0)
            out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
            img_pred = out_pred.get_image()[:, :, ::-1]
            processed_pred_images.append(img_pred) # Resize later in helper

        if not processed_gt_images:
            print("❌ ERROR: No images were successfully processed for grids.")
            return

        # --- Stitch and Save Grids ---
        print("Stitching grids...")
        gt_grid_image = create_image_grid(processed_gt_images, GRID_ROWS, GRID_COLS, IMG_HEIGHT, IMG_WIDTH)
        pred_grid_image = create_image_grid(processed_pred_images, GRID_ROWS, GRID_COLS, IMG_HEIGHT, IMG_WIDTH)

        gt_save_path = os.path.join(output_viz_dir, f"{dataset_name}_ground_truth_4x4.jpg")
        pred_save_path = os.path.join(output_viz_dir, f"{dataset_name}_prediction_4x4_conf{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}.jpg")

        cv2.imwrite(gt_save_path, gt_grid_image)
        cv2.imwrite(pred_save_path, pred_grid_image)

        print(f"✅ Ground Truth 4x4 grid saved to: '{gt_save_path}'")
        print(f"✅ Prediction 4x4 grid saved to:   '{pred_save_path}'")
            
    except Exception as e:
        print(f"❌ ERROR during grid visualization: {e}")
            
    print("\n--- Full Evaluation and Grid-Saving Complete ---")


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
        cfg_eval_std = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh, is_eval=True) 
        if std_test_dataset_name in DatasetCatalog.list():
             if metadata_std is None: metadata_std = MetadataCatalog.get(std_test_dataset_name)
             evaluate_test_set(cfg_eval_std, std_test_dataset_name, metadata_std) 
        else:
             print("❌ ERROR: Standard test dataset was not registered successfully. Cannot run evaluation.")
             
    elif choice == '3': 
        cfg_eval_davao = setup_cfg(config_path, weights_path, confidence_threshold=conf_thresh, is_eval=True)
        if davao_test_dataset_name in DatasetCatalog.list():

            if metadata_std is not None:
                print("Using standard metadata map for Davao evaluation.")
                # --- THIS LINE IS CHANGED ---
                evaluate_and_save_grids(cfg_eval_davao, davao_test_dataset_name, metadata_std) # Pass metadata_std
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


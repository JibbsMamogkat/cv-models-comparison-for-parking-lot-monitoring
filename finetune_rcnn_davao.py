# --- Standard Python Libraries ---
import os
import random
import cv2 
import torch
import numpy as np
import sys

# --- Detectron2 Libraries ---
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def main():
    """
    Main function to FINE-TUNE a BASE COCO model on the Davao dataset.
    This script runs Experiment B and MATCHES the groupmate's YOLO split.
    """
    # --- 1. SETUP LOGGING AND PATHS ---
    setup_logger()
    print("--- Starting R-CNN FINE-TUNING on Davao Dataset (from COCO) ---")
    
    json_split_dir = os.path.join("data", "processed", "davao_finetune_set_JSON_MATCHED")
    finetune_json_path = os.path.join(json_split_dir, "davao_finetune_train_MATCHED.json")
    valid_json_path = os.path.join(json_split_dir, "davao_finetune_valid_MATCHED.json")
    
  
    davao_image_dir = os.path.join("data", "real_world_test_set")
    
    output_dir = "./output_davao_finetuned_from_COCO"
    
    # --- 2. REGISTER NEW DATASETS ---
    print(f"Registering fine-tune datasets from: {json_split_dir}")
    
    train_dataset_name = "davao_finetune_train_matched"
    valid_dataset_name = "davao_finetune_valid_matched"
    
    try:
        if train_dataset_name not in DatasetCatalog.list():
            register_coco_instances(train_dataset_name, {}, finetune_json_path, davao_image_dir)
        if valid_dataset_name not in DatasetCatalog.list():
            register_coco_instances(valid_dataset_name, {}, valid_json_path, davao_image_dir)
        print("✅ Fine-tuning datasets registered successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not register datasets. Error: {e}")
        print("   Did you run 'match_yolo_split.py' first?")
        return

    # --- 3. CONFIGURE THE MODEL FOR FINE-TUNING ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 4 
    
    print("Setting NUM_CLASSES to 3 (0: spaces, 1: space-empty, 2: space-occupied)")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    
    cfg.SOLVER.IMS_PER_BATCH = 2 
    
    # --- Set training parameters ---
    cfg.SOLVER.BASE_LR = 0.00025 
    cfg.SOLVER.MAX_ITER = 3000     
    cfg.SOLVER.STEPS = (2000, 2500) 
    cfg.SOLVER.GAMMA = 0.1
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  
    
    # Set the new output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("\n✅ Fine-tuning configuration loaded.")
    print(f"   - Starting from model: ORIGINAL COCO MODEL")
    print(f"   - Training on: '{train_dataset_name}'")
    print(f"   - Saving new model to: '{cfg.OUTPUT_DIR}'")

    # --- 4. TRAIN THE MODEL ---
    trainer = DefaultTrainer(cfg)
    
    trainer.resume_or_load(resume=True) 
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()
    print("\n--- Fine-Tuning Complete! ---")

        # --- 5. EVALUATE THE *NEW* MODEL ---
    print("\n--- Starting Evaluation on Davao Validation Set ---")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    eval_metadata = MetadataCatalog.get("__temp_eval_meta")
    eval_metadata.thing_classes = ["space-empty", "space-occupied"]
    eval_metadata.thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}
    print("Manually creating 2-class metadata for evaluation.")

    # --- Run evaluation ---
    evaluator = COCOEvaluator(valid_dataset_name, output_dir=cfg.OUTPUT_DIR, tasks=("bbox",))
    evaluator._metadata = eval_metadata  

    test_loader = build_detection_test_loader(cfg, valid_dataset_name)

    try:
        results = inference_on_dataset(trainer.model, test_loader, evaluator)
        print(f"\n--- Fine-Tuned Evaluation Results ---")
        print(results)
    except Exception as e:
        print(f"❌ ERROR during evaluation: {e}")

    print("--- Evaluation Complete ---")

   # --- 6. SAVE PREDICTION VISUALIZATIONS (AS A BATCH) ---
    print("\n--- Saving Predictions on Validation Images (as a Batch) ---")

    GRID_ROWS = 4
    GRID_COLS = 3
    IMG_WIDTH = 640 
    IMG_HEIGHT = 640
    BATCH_OUTPUT_FILE = os.path.join(cfg.OUTPUT_DIR, "prediction_batch.jpg")

    predictor = DefaultPredictor(cfg)
    if 'dataset_dicts' not in locals():
        dataset_dicts = DatasetCatalog.get(valid_dataset_name)

    if 'eval_metadata' not in locals():
        eval_metadata = MetadataCatalog.get("__temp_eval_meta")
        eval_metadata.thing_classes = ["space-empty", "space-occupied"]
        eval_metadata.thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}

    processed_images = []

    for d in dataset_dicts[:(GRID_ROWS * GRID_COLS)]: 
        img_path = d["file_name"]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Skipping missing image for PRED batch: {img_path}")
            blank_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank_img, "Image Missing", (50, IMG_HEIGHT // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed_images.append(blank_img)
            continue

        outputs = predictor(img)

        v = Visualizer(
            img[:, :, ::-1],
            metadata=eval_metadata,
            scale=1.0 
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))


        img_with_preds = out.get_image()[:, :, ::-1]
        img_resized = cv2.resize(img_with_preds, (IMG_WIDTH, IMG_HEIGHT))
        processed_images.append(img_resized)
    
    if not processed_images:
        print("Error: No images were processed for the prediction batch.")
    else:
    
        num_needed = GRID_ROWS * GRID_COLS
        while len(processed_images) < num_needed:
            print("Adding blank image to fill prediction grid...")
            blank_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            processed_images.append(blank_img)

        rows_list = []
        for i in range(GRID_ROWS):
            start_index = i * GRID_COLS
            end_index = start_index + GRID_COLS
            img_row = processed_images[start_index:end_index]
            h_stacked_row = np.hstack(img_row)
            rows_list.append(h_stacked_row)
        
        final_batch_image = np.vstack(rows_list)
        
        cv2.imwrite(BATCH_OUTPUT_FILE, final_batch_image)
        print(f"\n✅✅✅ Successfully saved PREDICTION batch image to: {BATCH_OUTPUT_FILE} ---")

    # --- 7. (NEW) SAVE GROUND TRUTH BATCH VISUALIZATION ---
    print("\n--- Saving Ground Truth Batch on Validation Images ---")

    GRID_ROWS = 4
    GRID_COLS = 3
    IMG_WIDTH = 640 
    IMG_HEIGHT = 640
    BATCH_OUTPUT_FILE = os.path.join(cfg.OUTPUT_DIR, "ground_truth_batch.jpg")

    processed_images = []

    if 'dataset_dicts' not in locals():
        dataset_dicts = DatasetCatalog.get(valid_dataset_name)

    if 'eval_metadata' not in locals():
        eval_metadata = MetadataCatalog.get("__temp_eval_meta")
        eval_metadata.thing_classes = ["space-empty", "space-occupied"]
        eval_metadata.thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}

    for d in dataset_dicts[:(GRID_ROWS * GRID_COLS)]: 
        img_path = d["file_name"]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Skipping missing image for GT batch: {img_path}")
            blank_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank_img, "Image Missing", (50, IMG_HEIGHT // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed_images.append(blank_img)
            continue

        v_gt = Visualizer(
            img[:, :, ::-1],
            metadata=eval_metadata, 
            scale=1.0
        )
        
        out = v_gt.draw_dataset_dict(d) 
        
        img_with_boxes = out.get_image()[:, :, ::-1]
        img_resized = cv2.resize(img_with_boxes, (IMG_WIDTH, IMG_HEIGHT))
        processed_images.append(img_resized)
    
    if not processed_images:
        print("Error: No images were processed for the ground truth batch.")
        return

    num_needed = GRID_ROWS * GRID_COLS
    while len(processed_images) < num_needed:
        print("Adding blank image to fill grid...")
        blank_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        processed_images.append(blank_img)

    rows_list = []
    for i in range(GRID_ROWS):
        start_index = i * GRID_COLS
        end_index = start_index + GRID_COLS
        img_row = processed_images[start_index:end_index]
        h_stacked_row = np.hstack(img_row)
        rows_list.append(h_stacked_row)
    
    final_batch_image = np.vstack(rows_list)

    cv2.imwrite(BATCH_OUTPUT_FILE, final_batch_image)
    print(f"\n✅✅✅ Successfully saved GROUND TRUTH batch image to: {BATCH_OUTPUT_FILE} ---")



if __name__ == "__main__":
    try:
        import tensorflow as tf
    except ImportError:
        print("Warning: TensorFlow not found. Classifier scripts might fail.")
        pass 
        
    try:
        import tqdm
    except ImportError:
        print("Warning: tqdm not found. Progress bars will not show.")
        pass 

    main()
